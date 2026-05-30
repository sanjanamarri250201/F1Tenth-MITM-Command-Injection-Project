#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import threading
import time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import casadi as ca

class MPCNodeIDS(object):

    # ── MPC parameters ───────────────────────────────────────────────
    N           = 8
    DT          = 0.05
    WHEELBASE   = 0.3302
    MAX_STEER   = 0.4189
    MAX_SPEED   = 2.0
    MIN_SPEED   = 0.3
    W_HEADING   = 10.0
    W_JERK      = 8.0
    W_STEER     = 2.0
    WALL_THRESH = 1.5
    GAP_BUBBLE  = 0.4

    # ── wall distance thresholds ─────────────────────────────────────
    # NORMAL    : wall > 0.5m  → green
    # WARNING   : 0.3-0.5m    → cyan  — MANUAL ATTENTION REQUIRED
    # COLLISION : wall < 0.3m → red   — COLLISION DETECTED (permanent)
    DIST_NORMAL  = 0.530
    DIST_WARNING = 0.225

    # ── IDS thresholds ───────────────────────────────────────────────
    # Turn suppression: MPC steer above this AND /drive shows straight
    # Set just above WARNING (0.3m) value so IDS fires before collision
    IDS_TURN_THRESH     = 0.18  # MPC steer above this = MPC wants to turn
    IDS_SUPPRESS_THRESH = 0.08  # /drive steer below this = suppressed
    IDS_SPEED_THRESH    = 0.8   # /drive speed this much above MPC = injection
    # ────────────────────────────────────────────────────────────────

    def __init__(self):
        rospy.init_node('mpc_node_ids', anonymous=True)

        self.scan        = None
        self.odom        = None
        self.prev_steer  = 0.0

        self.display_scan        = None
        self.display_speed       = 0.0
        self.display_steer       = 0.0
        self.display_x           = 0.0
        self.display_y           = 0.0
        self.display_yaw         = 0.0
        self.display_mpc_steer   = 0.0
        self.display_mpc_speed   = 0.0
        self.display_drive_steer = 0.0
        self.display_drive_speed = 0.0
        self.min_wall_dist       = 99.0

        self.status      = 'NORMAL'
        self.status_time = None

        self.ids_alert      = False
        self.ids_alert_type = ''
        self.ids_time       = None
        self.ids_expected   = 0.0
        self.ids_received   = 0.0

        self.lock = threading.Lock()

        self._build_mpc()

        self.drive_pub = rospy.Publisher(
            '/drive', AckermannDriveStamped, queue_size=1)

        rospy.Subscriber('/scan',  LaserScan,
                         self._scan_cb,  queue_size=1)
        rospy.Subscriber('/odom',  Odometry,
                         self._odom_cb,  queue_size=1)
        rospy.Subscriber('/drive', AckermannDriveStamped,
                         self._drive_cb, queue_size=1)

        self.viz_thread = threading.Thread(target=self._viz_loop)
        self.viz_thread.daemon = True
        self.viz_thread.start()

        rospy.loginfo("MPC IDS node ready")
        rospy.spin()

    # ── callbacks ─────────────────────────────────────────────────────

    def _scan_cb(self, msg):
        self.scan = msg
        valid    = [r for r in msg.ranges
                    if not (np.isnan(r) or np.isinf(r)) and r > 0.01]
        min_wall = min(valid) if valid else 99.0

        with self.lock:
            self.display_scan  = msg
            self.min_wall_dist = min_wall

            if self.status != 'COLLISION':
                if min_wall < self.DIST_WARNING:
                    self.status      = 'COLLISION'
                    self.status_time = time.time()
                    rospy.logerr("COLLISION — wall at %.3fm", min_wall)

                elif min_wall <= self.DIST_NORMAL:
                    if self.status == 'NORMAL':
                        self.status      = 'WARNING'
                        self.status_time = time.time()
                        rospy.logwarn("WARNING — wall at %.3fm", min_wall)

                else:
                    if self.status == 'WARNING':
                        self.status = 'NORMAL'

        if self.odom is not None:
            self._run_mpc()

    def _odom_cb(self, msg):
        self.odom = msg
        q    = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = np.arctan2(siny, cosy)
        vx   = msg.twist.twist.linear.x
        vy   = msg.twist.twist.linear.y
        with self.lock:
            self.display_x     = msg.pose.pose.position.x
            self.display_y     = msg.pose.pose.position.y
            self.display_yaw   = yaw
            self.display_speed = np.sqrt(vx**2 + vy**2)

    def _drive_cb(self, msg):
        incoming_steer = msg.drive.steering_angle
        incoming_speed = msg.drive.speed

        with self.lock:
            mpc_steer = self.display_mpc_steer
            mpc_speed = self.display_mpc_speed
            self.display_drive_steer = incoming_steer
            self.display_drive_speed = incoming_speed

            # Rule 1 — turn suppression
            # IDS_TURN_THRESH = 0.18 — just above DIST_WARNING (0.3m)
            # so IDS fires at turns that would bring car into warning zone
            mpc_turning      = abs(mpc_steer)      > self.IDS_TURN_THRESH
            drive_suppressed = abs(incoming_steer) < self.IDS_SUPPRESS_THRESH

            if mpc_turning and drive_suppressed:
                if not self.ids_alert:
                    self.ids_alert      = True
                    self.ids_alert_type = 'TURN SUPPRESSION'
                    self.ids_time       = time.time()
                    self.ids_expected   = mpc_steer
                    self.ids_received   = incoming_steer
                    rospy.logwarn(
                        "[IDS] TURN SUPPRESSION — "
                        "expected=%.3f  received=%.3f",
                        mpc_steer, incoming_steer)

            # Rule 2 — speed injection
            elif incoming_speed > mpc_speed + self.IDS_SPEED_THRESH:
                if not self.ids_alert:
                    self.ids_alert      = True
                    self.ids_alert_type = 'SPEED INJECTION'
                    self.ids_time       = time.time()
                    self.ids_expected   = mpc_speed
                    self.ids_received   = incoming_speed
                    rospy.logwarn(
                        "[IDS] SPEED INJECTION — "
                        "expected=%.2f  received=%.2f",
                        mpc_speed, incoming_speed)

            else:
                if self.status == 'NORMAL':
                    self.ids_alert = False

    # ── gap detection ─────────────────────────────────────────────────

    def _find_best_gap(self, ranges, angle_min, angle_inc):
        ranges = np.array(ranges, dtype=np.float32)
        ranges = np.clip(ranges, 0.0, 10.0)
        ranges = np.where(np.isnan(ranges) | np.isinf(ranges), 0.0, ranges)

        proc = ranges.copy()
        for i in range(len(proc)):
            if 0 < proc[i] < self.WALL_THRESH:
                bubble = int(np.arctan2(self.GAP_BUBBLE, proc[i]) / angle_inc)
                bubble = max(1, min(bubble, 30))
                lo = max(0, i - bubble)
                hi = min(len(proc), i + bubble + 1)
                proc[lo:hi] = 0.0

        gaps  = []
        start = None
        for i in range(len(proc)):
            if proc[i] > 0:
                if start is None:
                    start = i
            else:
                if start is not None:
                    gaps.append((start, i - 1))
                    start = None
        if start is not None:
            gaps.append((start, len(proc) - 1))

        if not gaps:
            best_idx  = int(np.argmax(ranges))
            gap_angle = angle_min + best_idx * angle_inc
            gap_dist  = float(ranges[best_idx])
            return np.clip(gap_angle,
                           -self.MAX_STEER, self.MAX_STEER), gap_dist

        best_score           = -1
        best_start, best_end = gaps[0]
        for (s, e) in gaps:
            width        = e - s
            avg_depth    = float(np.mean(proc[s:e+1]))
            centre_idx   = (s + e) / 2.0
            centre_angle = angle_min + centre_idx * angle_inc
            forward_bias = np.exp(-4.0 * centre_angle**2)
            score        = width * avg_depth * forward_bias
            if score > best_score:
                best_score = score
                best_start = s
                best_end   = e

        best_idx  = (best_start + best_end) // 2
        gap_angle = angle_min + best_idx * angle_inc
        gap_dist  = float(proc[best_idx]) if proc[best_idx] > 0 else 0.5
        return np.clip(gap_angle,
                       -self.MAX_STEER, self.MAX_STEER), gap_dist

    # ── MPC build ─────────────────────────────────────────────────────

    def _build_mpc(self):
        N = self.N
        U = ca.MX.sym('U', N)
        P = ca.MX.sym('P', 2)
        cost = 0
        for k in range(N):
            cost += self.W_HEADING * (U[k] - P[0])**2
            cost += self.W_STEER   * U[k]**2
            prev  = P[1] if k == 0 else U[k-1]
            cost += self.W_JERK * (U[k] - prev)**2
        nlp  = {'x': U, 'f': cost, 'p': P}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter':    20,
            'ipopt.tol':         1e-3,
            'print_time':        0,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.u0     = np.zeros(N)
        rospy.loginfo("CasADi MPC IDS solver built (horizon=%d, dt=%.2f)",
                      N, self.DT)

    # ── MPC loop ──────────────────────────────────────────────────────

    def _run_mpc(self):
        scan = self.scan
        odom = self.odom

        q    = odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = np.arctan2(siny, cosy)
        vx   = odom.twist.twist.linear.x
        vy   = odom.twist.twist.linear.y

        gap_angle, _ = self._find_best_gap(
            scan.ranges, scan.angle_min, scan.angle_increment)

        valid    = [r for r in scan.ranges
                    if not (np.isnan(r) or np.isinf(r)) and r > 0.01]
        min_wall = min(valid) if valid else 1.0

        p_val = np.array([gap_angle, self.prev_steer])
        lbx   = [-self.MAX_STEER] * self.N
        ubx   = [ self.MAX_STEER] * self.N

        try:
            sol   = self.solver(x0=self.u0, p=p_val, lbx=lbx, ubx=ubx)
            u_opt = np.array(sol['x']).flatten()
            self.u0 = u_opt
            steer   = float(np.clip(u_opt[0],
                                    -self.MAX_STEER, self.MAX_STEER))
        except Exception as e:
            rospy.logwarn("MPC solve failed: %s", str(e))
            steer = 0.0

        speed_factor = 1.0 - 0.7 * (abs(steer) / self.MAX_STEER)
        wall_factor  = min(1.0, max(0.2, (min_wall - 0.3) / 1.2))
        speed        = self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * \
                       speed_factor * wall_factor

        self.prev_steer = steer

        with self.lock:
            self.display_mpc_steer = steer
            self.display_mpc_speed = speed
            self.display_steer     = steer

        msg = AckermannDriveStamped()
        msg.header.stamp         = rospy.Time.now()
        msg.header.frame_id      = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed          = speed
        self.drive_pub.publish(msg)

    # ── OpenCV viz ────────────────────────────────────────────────────

    def _viz_loop(self):
        WIN_W, WIN_H = 900, 560
        LIDAR_W      = 500
        CX, CY       = LIDAR_W // 2, WIN_H // 2
        SCALE        = 40

        cv2.namedWindow('F1Tenth MPC IDS Monitor', cv2.WINDOW_AUTOSIZE)

        while not rospy.is_shutdown():
            frame = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

            with self.lock:
                scan        = self.display_scan
                speed       = self.display_speed
                steer       = self.display_steer
                x           = self.display_x
                y           = self.display_y
                yaw         = self.display_yaw
                mpc_steer   = self.display_mpc_steer
                mpc_speed   = self.display_mpc_speed
                drv_steer   = self.display_drive_steer
                drv_speed   = self.display_drive_speed
                min_wall    = self.min_wall_dist
                status      = self.status
                stime       = self.status_time
                ids_alert   = self.ids_alert
                ids_type    = self.ids_alert_type
                ids_time    = self.ids_time
                ids_exp     = self.ids_expected
                ids_recv    = self.ids_received

            elapsed     = (time.time() - stime)    if stime    else 0
            ids_elapsed = (time.time() - ids_time) if ids_time else 0
            pulse       = abs(np.sin(time.time() * 4))

            # ── LEFT PANEL ──────────────────────────────────────────
            cv2.rectangle(frame, (0, 0), (LIDAR_W, WIN_H),
                          (15, 15, 15), -1)

            for r in [1, 2, 3, 4, 5]:
                cv2.circle(frame, (CX, CY), r * SCALE,
                           (40, 40, 40), 1)
                cv2.putText(frame, '{}m'.format(r),
                            (CX + r * SCALE + 2, CY - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (60, 60, 60), 1)
            cv2.line(frame, (CX, 0),  (CX, WIN_H),   (35, 35, 35), 1)
            cv2.line(frame, (0,  CY), (LIDAR_W, CY), (35, 35, 35), 1)

            if scan is not None:
                ranges = np.array(scan.ranges)
                for i in range(0, len(ranges), 3):
                    r = ranges[i]
                    if np.isnan(r) or np.isinf(r) or r <= 0:
                        continue
                    r = min(r, 5.0)
                    angle = scan.angle_min + i * scan.angle_increment
                    px = int(CX - r * SCALE * np.sin(angle))
                    py = int(CY - r * SCALE * np.cos(angle))
                    if r > self.DIST_NORMAL:
                        col = (0, 200, 0)
                    elif r > self.DIST_WARNING:
                        col = (0, 200, 200)
                    else:
                        col = (0, 0, 255)
                    cv2.circle(frame, (px, py), 2, col, -1)

            cv2.arrowedLine(frame, (CX, CY), (CX, CY - 25),
                            (255, 255, 255), 2, tipLength=0.4)
            cv2.circle(frame, (CX, CY), 6, (255, 255, 255), -1)
            cv2.ellipse(frame, (CX, CY), (20, 20), -90,
                        0, int(np.degrees(steer * 60)),
                        (100, 100, 255), 2)
            cv2.putText(frame, 'LIDAR VIEW', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (120, 120, 120), 1)

            # legend
            cv2.putText(frame, '>0.5m  NORMAL',
                        (10, WIN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (0, 200, 0), 1)
            cv2.putText(frame, '0.3-0.5m  WARNING — MANUAL ATTENTION',
                        (10, WIN_H - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (0, 200, 200), 1)
            cv2.putText(frame, '<0.3m  COLLISION DETECTED',
                        (10, WIN_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (0, 0, 255), 1)

            # wall bar
            bar_w = int(np.clip(min_wall / 5.0, 0, 1) * (LIDAR_W - 20))
            if min_wall > self.DIST_NORMAL:
                bar_col = (0, 200, 0)
            elif min_wall > self.DIST_WARNING:
                bar_col = (0, 200, 200)
            else:
                bar_col = (0, 0, 255)
            cv2.rectangle(frame, (10, WIN_H - 15),
                          (10 + bar_w, WIN_H - 8), bar_col, -1)
            cv2.putText(frame,
                        'nearest wall: {:.3f}m'.format(min_wall),
                        (10, WIN_H - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (120, 120, 120), 1)

            # wall status overlays
            if status == 'WARNING':
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (LIDAR_W, WIN_H),
                              (0, 120, 120), -1)
                cv2.addWeighted(ov, 0.10 * pulse, frame,
                                1 - 0.10 * pulse, 0, frame)
                cv2.putText(frame, 'WALL PROXIMITY WARNING',
                            (28, CY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (0, 200, 200), 2)
                cv2.putText(frame, 'MANUAL ATTENTION REQUIRED',
                            (22, CY + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (0, 180, 180), 1)
                cv2.putText(frame,
                            'wall: {:.3f}m'.format(min_wall),
                            (28, CY + 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 160, 160), 1)

            elif status == 'COLLISION':
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (LIDAR_W, WIN_H),
                              (0, 0, 180), -1)
                cv2.addWeighted(ov, 0.25 * pulse, frame,
                                1 - 0.25 * pulse, 0, frame)
                cv2.putText(frame, 'COLLISION DETECTED',
                            (35, CY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.72, (0, 60, 255), 2)
                cv2.putText(frame,
                            'wall: {:.3f}m  T+{:.1f}s'.format(
                                min_wall, elapsed),
                            (35, CY + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (80, 80, 255), 1)

            # ── RIGHT PANEL ─────────────────────────────────────────
            ox = LIDAR_W + 10
            cv2.rectangle(frame, (LIDAR_W, 0), (WIN_W, WIN_H),
                          (10, 10, 10), -1)
            cv2.line(frame, (LIDAR_W, 0), (LIDAR_W, WIN_H),
                     (60, 60, 60), 1)

            def txt(label, value, row, color=(200, 200, 200)):
                cv2.putText(frame,
                            '{}: {}'.format(label, value),
                            (ox, 40 + row * 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, color, 1)

            cv2.putText(frame, 'TELEMETRY', (ox, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (120, 120, 120), 1)
            txt('Speed   ', '{:.2f} m/s'.format(speed),      0)
            txt('Steer   ', '{:.3f} rad'.format(steer),       1)
            txt('Pos X   ', '{:.2f} m'.format(x),             2)
            txt('Pos Y   ', '{:.2f} m'.format(y),             3)
            txt('Heading ', '{:.1f} deg'.format(
                             np.degrees(yaw)),                 4)

            cv2.line(frame,
                     (LIDAR_W, 40 + 5 * 32),
                     (WIN_W,   40 + 5 * 32), (50, 50, 50), 1)

            cv2.putText(frame, 'MPC OUTPUT',
                        (ox, 40 + 5 * 32 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (100, 100, 180), 1)
            txt('MPC steer', '{:.3f}'.format(mpc_steer),
                6, (150, 150, 255))
            txt('MPC speed', '{:.2f}'.format(mpc_speed),
                7, (150, 150, 255))

            cv2.putText(frame, '/drive TOPIC',
                        (ox, 40 + 8 * 32 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (100, 180, 100), 1)

            drv_col = (0, 80, 255) if ids_alert else (100, 220, 100)
            txt('drv steer', '{:.3f}'.format(drv_steer),
                9,  drv_col)
            txt('drv speed', '{:.2f}'.format(drv_speed),
                10, drv_col)

            # ── IDS PANEL ───────────────────────────────────────────
            ids_y = 40 + 11 * 32
            cv2.line(frame, (LIDAR_W, ids_y - 8),
                     (WIN_W,  ids_y - 8), (50, 50, 50), 1)

            if ids_alert:
                cv2.rectangle(frame,
                              (LIDAR_W + 5, ids_y),
                              (WIN_W - 5,   WIN_H - 40),
                              (0, 0, 100), -1)
                cv2.rectangle(frame,
                              (LIDAR_W + 5, ids_y),
                              (WIN_W - 5,   WIN_H - 40),
                              (0, 0, 220), 1)
                cv2.putText(frame, 'IDS ALERT',
                            (ox, ids_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 60, 255), 2)
                cv2.putText(frame, ids_type,
                            (ox, ids_y + 44),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 80, 255), 1)
                cv2.putText(frame,
                            'expected: {:.3f}'.format(ids_exp),
                            (ox, ids_y + 64),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 100, 255), 1)
                cv2.putText(frame,
                            'received: {:.3f}'.format(ids_recv),
                            (ox, ids_y + 84),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 100, 255), 1)
                cv2.putText(frame,
                            'delta:    {:.3f}'.format(
                                abs(ids_exp - ids_recv)),
                            (ox, ids_y + 104),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 120, 255), 1)
                cv2.putText(frame,
                            'T+{:.1f}s'.format(ids_elapsed),
                            (ox, ids_y + 124),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (80, 80, 255), 1)
            else:
                cv2.putText(frame, 'IDS: MONITORING',
                            (ox, ids_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 160, 0), 1)
                cv2.putText(frame,
                            'expected: {:.3f}'.format(mpc_steer),
                            (ox, ids_y + 44),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (80, 160, 80), 1)
                cv2.putText(frame,
                            'received: {:.3f}'.format(drv_steer),
                            (ox, ids_y + 64),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (80, 160, 80), 1)
                cv2.putText(frame,
                            'delta:    {:.3f}'.format(
                                abs(mpc_steer - drv_steer)),
                            (ox, ids_y + 84),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (80, 160, 80), 1)

            # status bar
            styles = {
                'NORMAL':    ((0, 60, 0),
                              (0, 200, 0),
                              'STATUS: NORMAL  (wall > 0.5m)'),
                'WARNING':   ((0, 80, 80),
                              (0, 200, 200),
                              'STATUS: WARNING  (0.3-0.5m) — MANUAL ATTENTION REQUIRED'),
                'COLLISION': ((0, 0, 100),
                              (0, 50, 255),
                              'STATUS: COLLISION DETECTED  (< 0.3m)'),
            }
            bg, fg, label = styles[status]
            if ids_alert:
                bg    = (60, 0, 80)
                fg    = (180, 0, 255)
                label = 'STATUS: {} | IDS: {}'.format(status, ids_type)
            cv2.rectangle(frame,
                          (LIDAR_W, WIN_H - 35),
                          (WIN_W, WIN_H), bg, -1)
            cv2.putText(frame, label, (ox, WIN_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, fg, 1)

            cv2.imshow('F1Tenth MPC IDS Monitor', frame)
            cv2.waitKey(50)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        MPCNodeIDS()
    except rospy.ROSInterruptException:
        pass


