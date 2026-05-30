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

class MPCNode(object):

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
    # WARNING   : 0.3m-0.5m   → cyan  — MANUAL ATTENTION REQUIRED
    # COLLISION : wall < 0.3m → red   — COLLISION DETECTED (permanent)
    DIST_NORMAL    = 0.530
    DIST_WARNING   = 0.225
    # ────────────────────────────────────────────────────────────────

    def __init__(self):
        rospy.init_node('mpc_node', anonymous=True)

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
        self.status              = 'NORMAL'
        self.status_time         = None
        self.lock                = threading.Lock()

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

        rospy.loginfo("MPC node ready")
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
                    # below 0.3m → COLLISION (permanent)
                    self.status      = 'COLLISION'
                    self.status_time = time.time()
                    rospy.logerr("COLLISION — wall at %.3fm", min_wall)

                elif min_wall <= self.DIST_NORMAL:
                    # 0.3m to 0.5m → WARNING
                    if self.status == 'NORMAL':
                        self.status      = 'WARNING'
                        self.status_time = time.time()
                        rospy.logwarn("WARNING — wall at %.3fm", min_wall)

                else:
                    # above 0.5m → NORMAL
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
        with self.lock:
            self.display_drive_steer = msg.drive.steering_angle
            self.display_drive_speed = msg.drive.speed

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
        rospy.loginfo("CasADi MPC solver built (horizon=%d, dt=%.2f)",
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

    def _draw_status_box(self, frame, text1, text2, text3,
                         box_col, border_col, txt_col,
                         CX, CY, LIDAR_W, WIN_H):
        cv2.rectangle(frame,
                      (20, CY - 60), (LIDAR_W - 20, CY + 60),
                      box_col, -1)
        cv2.rectangle(frame,
                      (20, CY - 60), (LIDAR_W - 20, CY + 60),
                      border_col, 2)
        if text1:
            cv2.putText(frame, text1, (35, CY - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_col, 2)
        if text2:
            cv2.putText(frame, text2, (35, CY + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_col, 1)
        if text3:
            cv2.putText(frame, text3, (35, CY + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_col, 1)

    def _viz_loop(self):
        WIN_W, WIN_H = 900, 500
        LIDAR_W      = 500
        CX, CY       = LIDAR_W // 2, WIN_H // 2
        SCALE        = 40

        cv2.namedWindow('F1Tenth MPC Monitor', cv2.WINDOW_AUTOSIZE)

        while not rospy.is_shutdown():
            frame = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

            with self.lock:
                scan       = self.display_scan
                speed      = self.display_speed
                steer      = self.display_steer
                x          = self.display_x
                y          = self.display_y
                yaw        = self.display_yaw
                mpc_steer  = self.display_mpc_steer
                mpc_speed  = self.display_mpc_speed
                drv_steer  = self.display_drive_steer
                drv_speed  = self.display_drive_speed
                min_wall   = self.min_wall_dist
                status     = self.status
                stime      = self.status_time

            elapsed = (time.time() - stime) if stime else 0
            pulse   = abs(np.sin(time.time() * 4))

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

            # LiDAR points
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
                        col = (0, 200, 0)       # green  — normal
                    elif r > self.DIST_WARNING:
                        col = (0, 200, 200)     # cyan   — warning
                    else:
                        col = (0, 0, 255)       # red    — collision zone
                    cv2.circle(frame, (px, py), 2, col, -1)

            # car arrow and steering arc
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

            # wall proximity bar
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

            # ── STATUS OVERLAYS ─────────────────────────────────────
            if status == 'WARNING':
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (LIDAR_W, WIN_H),
                              (0, 140, 140), -1)
                cv2.addWeighted(ov, 0.10 * pulse, frame,
                                1 - 0.10 * pulse, 0, frame)
                self._draw_status_box(
                    frame,
                    'WALL PROXIMITY WARNING',
                    'MANUAL ATTENTION REQUIRED',
                    'wall: {:.3f}m  (0.3-0.5m zone)'.format(min_wall),
                    (0, 100, 100), (0, 200, 200),
                    (0, 220, 220), CX, CY, LIDAR_W, WIN_H)

            elif status == 'COLLISION':
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (LIDAR_W, WIN_H),
                              (0, 0, 200), -1)
                cv2.addWeighted(ov, 0.25 * pulse, frame,
                                1 - 0.25 * pulse, 0, frame)
                self._draw_status_box(
                    frame,
                    'COLLISION DETECTED',
                    'wall: {:.3f}m  T+{:.1f}s'.format(
                        min_wall, elapsed),
                    '',
                    (0, 0, 140), (0, 0, 255),
                    (0, 60, 255), CX, CY, LIDAR_W, WIN_H)

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
                            0.55, color, 1)

            cv2.putText(frame, 'TELEMETRY', (ox, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (120, 120, 120), 1)
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
                        0.45, (100, 100, 180), 1)
            txt('MPC steer', '{:.3f}'.format(mpc_steer),
                6, (150, 150, 255))
            txt('MPC speed', '{:.2f}'.format(mpc_speed),
                7, (150, 150, 255))

            cv2.putText(frame, '/drive TOPIC',
                        (ox, 40 + 8 * 32 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (100, 180, 100), 1)
            txt('drv steer', '{:.3f}'.format(drv_steer),
                9,  (100, 220, 100))
            txt('drv speed', '{:.2f}'.format(drv_speed),
                10, (100, 220, 100))

            # status bar
            styles = {
                'NORMAL':    ((0, 60, 0),
                              (0, 200, 0),
                              'STATUS: NORMAL  (wall > 0.5m)'),
                'WARNING':   ((0, 90, 90),
                              (0, 220, 220),
                              'STATUS: WARNING  (0.3-0.5m) — MANUAL ATTENTION REQUIRED'),
                'COLLISION': ((0, 0, 120),
                              (0, 60, 255),
                              'STATUS: COLLISION DETECTED  (< 0.3m)'),
            }
            bg, fg, label = styles[status]
            cv2.rectangle(frame,
                          (LIDAR_W, WIN_H - 35),
                          (WIN_W, WIN_H), bg, -1)
            cv2.putText(frame, label, (ox, WIN_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, fg, 1)

            cv2.imshow('F1Tenth MPC Monitor', frame)
            cv2.waitKey(50)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        MPCNode()
    except rospy.ROSInterruptException:
        pass
