# image,rbbox,block,pixel
import matplotlib.pyplot as plt
time = [0.5, 5.46,15,30.3]
ap = [84.6,87.1,87.6,78.6]
pro= [98,98.5,98.4,96.1]
pixel_auroc = [99.4,99.7,99.7,99.0]
image_auroc = [99.7,99.8,99.8,99.4]
linewidth=1.5
plt.figure(figsize = (5.8,4))
colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.plot(time,ap,label='AP',color=colors[0],linewidth=linewidth)
plt.plot(time,pro,label='PRO',color=colors[1],linestyle=':',linewidth=linewidth)
plt.plot(time,image_auroc,label='Image AUROC',color=colors[2],linestyle='dashdot',linewidth=linewidth)
plt.plot(time,pixel_auroc,label='Pixel AUROC',color=colors[3],linestyle='--',linewidth=linewidth)

size=80

plt.scatter([time[0]]*4,[ap[0],pro[0],pixel_auroc[0],image_auroc[0]],marker='o',label='Ours-Weak-Sup (Image)',s=size,
            c=colors[:4],alpha=1,zorder=2)
plt.scatter([time[1]]*4,[ap[1],pro[1],pixel_auroc[1],image_auroc[1]],marker='P',label='Ours-Weak-Sup (RBBox)',s=size,
            c=colors[:4], alpha=1, zorder=2)
plt.scatter([time[2]]*4,[ap[2],pro[2],pixel_auroc[2],image_auroc[2]],marker='^',label='Ours-Weak-Sup (Block)',s=size,
            c=colors[:4], alpha=1, zorder=2)
plt.scatter([time[3]]*4,[ap[3],pro[3],pixel_auroc[3],image_auroc[3]],marker='h',label='PRN-Pixel-Sup',s=size,
            c=colors[:4], alpha=1, zorder=2)
plt.legend(ncol=2,fontsize=8.2)
plt.grid(True)
plt.xlabel("Annotation Speed (seconds/image)",fontsize=14)
plt.ylabel("AD Performance",fontsize=14)
plt.savefig('annotation_time.pdf',bbox_inches='tight')