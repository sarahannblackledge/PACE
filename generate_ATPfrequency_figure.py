import matplotlib.pyplot as plt

#Hard code frequency counts for each criteria
#A = [1, 2, 2, 4, 2]
#B = [6, 3, 7, 7, 3]
#C = [7, 4, 8, 8, 4]

#Hard code percent counts for each criteria
A = [1.72, 3.45, 3.45, 6.90, 3.45]
B = [10.34, 5.17, 12.10, 12.10, 5.17]
C = [12.07, 6.90, 13.80, 13.80, 6.90]


fraction = [1, 2, 3, 4, 5]

#Make bar plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
fig.suptitle('DPR plan acceptability by fraction', fontsize=20)
#Criteria A
ax1.bar(fraction, A, width=1, edgecolor='black')
ax1.set_title('Criteria A (19%) ', fontsize=14)
ax1.set_xlabel('Fraction number', fontsize=14)
ax1.set_ylabel('Percentage of fractions with acceptable plan in DPR', fontsize=14)

#Criteria B
ax2.bar(fraction, B, width=1, edgecolor='black')
ax2.set_title('Criteria B (45%)', fontsize=14)
ax2.set_xlabel('Fraction number', fontsize=14)

#Criteria C
ax3.bar(fraction, C, width=1, edgecolor='black')
ax3.set_title('Criteria C (54%)', fontsize=14)
ax3.set_xlabel('Fraction number', fontsize=14)

plt.show()

