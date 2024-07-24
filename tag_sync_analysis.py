width = 0.35  # the width of the bars

fig, (ax1,ax2) = plt.subplots(2,1)
#rects1 = ax1.bar(early_edges[1:] - width/2, early, width, label='Fine')
#rects2 = ax2.bar(late_edges[1:] + width/2, late, width, label='Checkerboard')

# Add some text for labels, title and custom x-axis tick labels, etc.

ax1.hist(early)
ax2.hist(late)
ax1.set_xlabel('Time [sec]')
ax1.set_title('TAG sync variations [sec]')

ax1.legend()
ax2.legend()
plt.show()