import matplotlib.pyplot as plt

# Step 1: Read FPS values from log file
with open("fps_log.txt", "r") as f:
    fps_values = [float(line.strip()) for line in f if line.strip()]

# Step 2: Generate Plot
plt.figure(figsize=(12, 5))
plt.plot(fps_values, color='blue', linewidth=1.5)
plt.xlabel("Frame Number")
plt.ylabel("FPS")
plt.title("FPS Over Time During Real-Time Detection")
plt.grid(True)
plt.tight_layout()

# Step 3: Save the plot as an image
plt.savefig("fps_plot.png")

# Optional: Display the plot
plt.show()
