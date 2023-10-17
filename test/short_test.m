

a = fullfile('~/wanglab/Data/Processed/2023-10-01-Jiahao-Test/mAD_64/');
b = dir(a);
folder_names = {b(:).name};
target_names = ["protein", "round2"];
c = b(ismember(folder_names, target_names));
c(:).name;
folder_names{:};
find({folder_names{:}}== "protein");

round_dir = dir(strcat(a, 'round*'))
round_dir.folder