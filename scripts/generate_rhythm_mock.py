import argparse
import math
import os
import random
import time

import numpy as np


CLASS_LABELS = [
    "ADL - Bending", "ADL - Lying/Rest", "ADL - Safe Activity",
    "ADL - Sitting", "ADL - Standing", "ADL - Walking",
    "FALL - Collapse", "FALL - Loss Balance", "FALL - Motionless",
    "FALL - Slow Slide", "FALL - Struggle", "FALL - Violent",
]


def smooth_one_hot(label, strength=0.86, noise=0.02):
    probs = np.full(12, (1.0 - strength) / 11.0, dtype=np.float32)
    probs[label] = strength
    probs += np.random.uniform(0.0, noise, size=12).astype(np.float32)
    probs /= probs.sum()
    return probs


def normal_label_for_slot(slot):
    hour = slot / 4.0
    if hour < 5.5:
        return random.choices([1, 8], weights=[0.97, 0.03])[0]
    if hour < 7.5:
        return random.choices([4, 5, 3], weights=[0.35, 0.45, 0.20])[0]
    if hour < 11.5:
        return random.choices([2, 4, 5, 0], weights=[0.35, 0.25, 0.30, 0.10])[0]
    if hour < 13.5:
        return random.choices([3, 1, 2], weights=[0.55, 0.30, 0.15])[0]
    if hour < 18.5:
        return random.choices([2, 4, 5, 0], weights=[0.40, 0.25, 0.25, 0.10])[0]
    if hour < 21.5:
        return random.choices([3, 2, 5], weights=[0.55, 0.30, 0.15])[0]
    return random.choices([1, 3, 4], weights=[0.65, 0.25, 0.10])[0]


def anomaly_label_for_slot(slot):
    hour = slot / 4.0
    if hour < 5.5:
        return random.choices([6, 8, 10, 5], weights=[0.25, 0.35, 0.25, 0.15])[0]
    return random.choices([6, 7, 8, 9, 10, 11], weights=[0.20, 0.18, 0.18, 0.12, 0.16, 0.16])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="core/data/rhythm_mock.npz")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    slots_per_day = (24 * 60) // args.step_minutes
    total_steps = args.days * slots_per_day
    base_time = int(time.time()) - (total_steps * args.step_minutes * 60)

    probs = np.zeros((total_steps, 12), dtype=np.float32)
    labels = np.zeros(total_steps, dtype=np.int64)
    anomaly = np.zeros(total_steps, dtype=np.int64)
    timestamps = np.zeros(total_steps, dtype=np.int64)
    time_features = np.zeros((total_steps, 2), dtype=np.float32)

    anomaly_slots = set()
    for day in range(args.days):
        for _ in range(random.randint(1, 3)):
            if random.random() < 0.45:
                slot = random.randint(0, 22)
            else:
                slot = random.randint(24, slots_per_day - 1)
            anomaly_slots.add(day * slots_per_day + slot)

    for idx in range(total_steps):
        slot = idx % slots_per_day
        timestamps[idx] = base_time + idx * args.step_minutes * 60
        angle = 2.0 * math.pi * slot / slots_per_day
        time_features[idx] = [math.sin(angle), math.cos(angle)]

        is_anomaly = idx in anomaly_slots
        label = anomaly_label_for_slot(slot) if is_anomaly else normal_label_for_slot(slot)
        labels[idx] = label
        anomaly[idx] = int(is_anomaly)
        probs[idx] = smooth_one_hot(label, strength=0.78 if is_anomaly else 0.88)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        probs=probs,
        labels=labels,
        anomaly=anomaly,
        timestamps=timestamps,
        time_features=time_features,
        class_labels=np.array(CLASS_LABELS),
        step_minutes=np.array(args.step_minutes),
        slots_per_day=np.array(slots_per_day),
    )
    print(f"saved {args.output}")
    print(f"steps={total_steps} anomalies={int(anomaly.sum())} step_minutes={args.step_minutes}")


if __name__ == "__main__":
    main()
