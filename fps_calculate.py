import matplotlib.pyplot as plt

# Load FPS values
with open("fps_log.txt", "r") as f:
    fps_values = [float(line.strip()) for line in f]

# Summary statistics
avg_fps = sum(fps_values) / len(fps_values)
min_fps = min(fps_values)
max_fps = max(fps_values)

print(f"Average FPS: {avg_fps:.2f}")
print(f"Minimum FPS: {min_fps:.2f}")
print(f"Maximum FPS: {max_fps:.2f}")
