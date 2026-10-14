#!/usr/bin/env python3
"""Print RAM and disk usage every few seconds with time-left estimates, and
auto-save the PCD map through pcd_map_saver_node before RAM runs out.

Started by odin1_ros2.launch.py (monitor:=true). Parameters:
  disk_path         directory whose filesystem is reported (default: home)
  period_sec        print interval (default 5)
  rate_window_sec   history used to estimate growth rates (default 60)
  ram_save_percent  RAM usage that triggers an auto-save (default 90)
  save_service      Trigger service that writes the PCD
"""
import os
import time
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

GB = 1024 ** 3


def read_ram():
    """Returns (used, total) bytes; used excludes reclaimable page cache."""
    info = {}
    with open('/proc/meminfo') as f:
        for line in f:
            key, value = line.split(':', 1)
            info[key] = int(value.split()[0]) * 1024
    return info['MemTotal'] - info['MemAvailable'], info['MemTotal']


def read_disk(path):
    """Returns (used, total) bytes for the filesystem holding path (or its nearest existing parent)."""
    while not os.path.exists(path) and path != os.path.dirname(path):
        path = os.path.dirname(path)
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    return total - st.f_bavail * st.f_frsize, total


def growth_rate(samples):
    """Least-squares slope in bytes/s over (t, used) samples, or None if too few."""
    if len(samples) < 3:
        return None
    n = len(samples)
    mt = sum(t for t, _ in samples) / n
    mu = sum(u for _, u in samples) / n
    var = sum((t - mt) ** 2 for t, _ in samples)
    if var == 0:
        return None
    return sum((t - mt) * (u - mu) for t, u in samples) / var


def time_left(remaining_bytes, rate):
    if rate is None:
        return 'estimating...'
    if remaining_bytes <= 0:
        return 'now'
    if rate <= 0:
        return 'not growing'
    minutes = remaining_bytes / rate / 60
    if minutes > 24 * 60:
        return '> 24 h'
    if minutes >= 60:
        return f'~{minutes / 60:.1f} h ({minutes:.0f} min)'
    return f'~{minutes:.0f} min'


class ResourceMonitor(Node):
    def __init__(self):
        super().__init__('resource_monitor')
        self.disk_path = self.declare_parameter('disk_path', os.path.expanduser('~')).value
        period = self.declare_parameter('period_sec', 5.0).value
        self.window = self.declare_parameter('rate_window_sec', 60.0).value
        self.save_percent = self.declare_parameter('ram_save_percent', 90.0).value
        service = self.declare_parameter('save_service', '/pcd_map_saver_node/save_pcd').value

        self.ram_hist = deque()
        self.disk_hist = deque()
        self.save_client = self.create_client(Trigger, service)
        self.save_armed = True  # re-armed once RAM drops 5% below the threshold
        self.saving = False
        self.create_timer(period, self.tick)
        self.get_logger().info(
            f'Monitoring RAM and disk ({self.disk_path}) every {period:.0f} s; '
            f'auto-save PCD at {self.save_percent:.0f}% RAM via {service}')

    def tick(self):
        now = time.monotonic()
        ram_used, ram_total = read_ram()
        disk_used, disk_total = read_disk(self.disk_path)
        for hist, used in ((self.ram_hist, ram_used), (self.disk_hist, disk_used)):
            hist.append((now, used))
            while hist and now - hist[0][0] > self.window:
                hist.popleft()

        ram_pct = 100.0 * ram_used / ram_total
        disk_pct = 100.0 * disk_used / disk_total
        ram_rate = growth_rate(self.ram_hist)
        disk_rate = growth_rate(self.disk_hist)
        save_at = ram_total * self.save_percent / 100.0

        ram_eta = f'RAM time left: {time_left(ram_total - ram_used, ram_rate)}'
        if self.save_armed and ram_used < save_at:
            ram_eta += f' (auto-save at {self.save_percent:.0f}% in {time_left(save_at - ram_used, ram_rate)})'
        self.get_logger().info(
            f'RAM: {ram_used / GB:.1f} / {ram_total / GB:.1f} GB ({ram_pct:.0f}%)   '
            f'Disk: {disk_used / GB:.0f} / {disk_total / GB:.0f} GB ({disk_pct:.0f}%)\n'
            f'    {ram_eta}\n'
            f'    Disk time left: {time_left(disk_total - disk_used, disk_rate)}')

        if ram_pct >= self.save_percent and self.save_armed:
            self.auto_save(ram_pct)
        elif ram_pct < self.save_percent - 5:
            self.save_armed = True

    def auto_save(self, ram_pct):
        stamp = datetime.now().strftime('%H:%M:%S on %A, %B %d, %Y')
        if not self.save_client.service_is_ready():
            self.get_logger().warn(
                f'RAM is about full ({ram_pct:.0f}%) at {stamp}, but {self.save_client.srv_name} '
                f'is not available (launched without pcd:=true?)')
            self.save_armed = False
            return
        if self.saving:
            return
        self.get_logger().warn(
            f'RAM is about full ({ram_pct:.0f}%), auto-saving PCD file at {stamp}. '
            f'Stop the run (Ctrl+C) soon; the map keeps growing after this save.')
        self.save_armed = False
        self.saving = True
        self.save_client.call_async(Trigger.Request()).add_done_callback(self.on_saved)

    def on_saved(self, future):
        self.saving = False
        stamp = datetime.now().strftime('%H:%M:%S on %Y-%m-%d')
        try:
            res = future.result()
        except Exception as e:  # service died mid-save
            self.get_logger().error(f'Auto-save failed at {stamp}: {e}')
            return
        if res.success:
            self.get_logger().warn(f'Auto-save finished at {stamp}: {res.message}')
        else:
            self.get_logger().error(f'Auto-save failed at {stamp}: {res.message}')


def main():
    rclpy.init()
    node = ResourceMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
