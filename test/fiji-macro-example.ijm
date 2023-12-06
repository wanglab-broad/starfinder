macro "batch_merge_channels"{
    setBatchMode(true);
    file1 = getDirectory("DAPI");
    list1 = getFileList(file1);

    file2 = getDirectory("actb");
    list2 = getFileList(file2); 
    
    file3 = getDirectory("Merge");
    list3 = getFileList(file3);
    
    n = 0;
    small = 2; 
    //condition for for-loop

    for(i = n + 1; i < small; i++) {
      //i will always follow the aborted number of merges, you might not have the problem, 
      //not to lose your track, though you can change it to anything else
      open(file1 + list1[i]);
      name = getTitle();
      open(file2 + list2[i]);
      name2 = "tile_10-1.tif";
      run("Merge Channels...", "c1=[" + name + "] c2=[" + name2 + "] create");
      saveAs("tiff", file3 + name);
    }
}