current_sample = "sample4"  
adata = sc.read_h5ad(os.path.join(expr_path, f'{current_sample}_raw.h5ad'))

# calculate pp metric
sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)

# Normalization scaling
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.raw = adata

# Scale data to unit variance and zero mean
sc.pp.scale(adata)
adata.layers['scaled'] = adata.X.copy()

adata

fig_size = np.array([adata.obs['global_x'].max(), adata.obs['global_y'].max()]) / 1000
plt.style.use("default")

# Plot the FOVs
fig, ax = plt.subplots(figsize=fig_size)
sns.scatterplot(x='global_x', y='global_y', hue='fov_id', data=adata.obs, s=1, legend=False, ax=ax)
plt.show()


# Plot total_counts
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=fig_size)
sns.scatterplot(x='global_x', y='global_y', hue='total_counts', data=adata.obs, s=1, palette='viridis', ax=ax)
plt.show()

positions = [d for d in os.listdir(expr_path) if "Position" in d]

total_assignment = []
total_coverage = []

for current_position in tqdm(positions):
    current_log = os.path.join(expr_path, current_position, 'log.txt')  
    with open(current_log, 'r') as f:
        lines = [line.rstrip() for line in f]
        current_assignment = lines[0].split('%')[0]
        current_coverage = lines[1].split(' ')[2].split('%')[0]
        total_assignment.append(float(current_assignment))
        total_coverage.append(float(current_coverage))

        if float(current_coverage) > 100:
            print(current_position, current_assignment)


plt.style.use("default")
sns.histplot(total_assignment, bins=20)
plt.show()