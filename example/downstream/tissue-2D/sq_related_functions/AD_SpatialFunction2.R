#' Main function for STARmap PLUS analysis
#'
#' Convert spatialExperiment object to a list of Seurat objects based on sample id
#'
#' @param starmap_obj Seurat Object contain one sample
#' @param img_list A list contain required images `plaque` for plaque analysis, `region(optional)` for region sub-setting, must have the same resolution as `plaque`
#' @param min_area Threshold for filtering small plaques
#' @param region Subset region
#' @param resolution_ micron per pixel
#'
#' @return Seurat Object w/ results
#'
#' @import Seurat
#' @import SpatialExperiment
#'
#' @export
#'
#' @examples
#' No Example
ad_plaque_quant <- function(starmap_obj,
                            img_list,
                            min_area = 400,
                            region = "all", resolution_ = 0.18){
  stopifnot("Inconsistent image resolution" = is.null(img_list[["region"]]) | all(dim(img_list[["plaque"]]) == dim(img_list[["region"]])))
  #Calc Plaque Center
  tic <- Sys.time()
  require(EBImage)
  message(paste(Sys.time(),"Processing Protein Image"))
  plaque <- Image(img_list[["plaque"]],colormode = Grayscale) %>% bwlabel()
  message(paste(Sys.time(),"Compute Plaque Features"))
  plaque_meta <- cbind(computeFeatures.moment(plaque),computeFeatures.shape(plaque)) %>%
    as.data.frame()  %>% data.frame(., plaque_id = as.numeric(row.names(.)))

  cell_meta <- data.frame(starmap_obj@meta.data,
                          min_center_dist = sqrt(sum(dim(plaque)^2)),
                          min_border_dist = sqrt(sum(dim(plaque)^2)),
                          nearest_plaque = 0,
                          interval = "other")

  #Organize cell_meta
  #orig_index |x  |y  |batch  |time |group  |top_level  |cell_type  |region |min_dist |
  #if(is.null(cell_meta$region)){
  #  cell_meta <-data.frame(orig_index = cell_meta$orig_index,
  #                         x = cell_meta$x, y = cell_meta$y,
  #                         batch = cell_meta$batch, time = cell_meta$time, group = cell_meta$group,
  #                         top_level = cell_meta$top_level, cell_type = cell_meta$cell_type,
  #                         region = NA,
  #                         nearest_plaque = 0, interval = "other",
  #                         min_center_dist = cell_meta$min_center_dist,
  #                         min_border_dist = cell_meta$min_border_dist)
  #}else{
  #  cell_meta <-data.frame(orig_index = cell_meta$orig_index,
  #                         x = cell_meta$x, y = cell_meta$y,
  #                         batch = cell_meta$batch, time = cell_meta$time, group = cell_meta$group,
  #                         top_level = cell_meta$top_level, cell_type = cell_meta$cell_type,
  #                         region = cell_meta$region,
  #                         nearest_plaque = 0, interval = "other",
  #                         min_center_dist = cell_meta$min_center_dist,
  #                         min_border_dist = cell_meta$min_border_dist)
  #}

  #Region Selection
  if(region != "all"){
    plaque_meta <- data.frame(plaque_meta,region = -1)
    plaque_meta$region <- apply(plaque_meta,1,
                                FUN = function(x){
                                  switch(img_list[["region"]][round(as.numeric(x[1])),round(as.numeric(x[2]))],
                                         '1' = "Cortex", '2' = "Corpus Callosum", '3' = "Hippocampus")
                                })
    if(sum(is.na(cell_meta$region)==F) == 0){
      cell_meta$region <- apply(cell_meta,1,
                                FUN = function(x){
                                  switch(img_list[["region"]][round(as.numeric(x[2])),round(as.numeric(x[3]))],
                                         '1' = "Cortex", '2' = "Corpus Callosum", '3' = "Hippocampus")
                                })
    }
  }
  #Save Raw palque meta for dilate
  plaque_meta_bk <- plaque_meta
  plaque_meta <- plaque_meta[plaque_meta$s.area > min_area,]

  if(region == "Sub-cortical"){
    cell_meta <- cell_meta[cell_meta$region %in% c("Hippocampus","Corpus Callosum"),]
    plaque_meta <- plaque_meta[plaque_meta$region %in% c("Hippocampus","Corpus Callosum"),]
  }else if(region != "all"){
    cell_meta <- cell_meta[cell_meta$region == region,]
    plaque_meta <- plaque_meta[plaque_meta$region == region,]
  }


  # Calc Plaque 5-round dilate area
  #message("Calc size of dilated areas")
  if(region == "Sub-cortical"){#Filter out
    plaque <- rmObjects(plaque,plaque_meta_bk$plaque_id[plaque_meta_bk$s.area <= min_area |
                                                          !(plaque_meta_bk$region %in% c("Hippocampus","Corpus Callosum"))],
                        reenumerate = F)
  }else if(region != "all"){
    plaque <- rmObjects(plaque,plaque_meta_bk$plaque_id[plaque_meta_bk$s.area <= min_area |
                                                          plaque_meta_bk$region != region], reenumerate = F)
  }else{
    plaque <- rmObjects(plaque,plaque_meta_bk$plaque_id[plaque_meta_bk$s.area <= min_area], reenumerate = F)
  }

  #Do voronoi
  voronoi_img <- propagate(seeds = plaque, x = Image(dim = dim(plaque)))

  # to dense matrix for fast palque assign
  voronoi_mat <- as.matrix(voronoi_img@.Data)

  cell_meta$nearest_plaque <- map2_vec(cell_meta$x, cell_meta$y,
                                       \(x,y) voronoi_mat[x,y],
                                       .progress = T)
  rm(voronoi_mat)
  gc()

  if(region == "all"){
    img_size <- dim(plaque)[1] * dim(plaque)[2]
  }else if(region == "Sub-cortical"){
    img_size <- sum(img_list[["region"]] == 2) +
      sum(img_list[["region"]] == 3)
    region_img <- Image(img_list[["region"]] == 2 |img_list[["region"]] == 3)
  }else{
    x <- switch(region,
                "Cortex" = 1,"Corpus Callosum" = 2, "Hippocampus" = 3)
    img_size <- sum(img_list[["region"]] == x)
    region_img <- Image(img_list[["region"]] == x)
  }

  #Assign interval
  message(paste(Sys.time(),"Compute Cell-Plaque Distance Interval"))
  dilate_ls <- list()
  size_vec  <- rep(0,5); names(size_vec) <- paste(1:5,"0um",sep = "")
  dilate_img <- plaque > 0
  dilate_img_bk <- dilate_img
  for(i in 1:5){
    message(paste("round:",i,"of 5"))
    dilate_img_bk <- dilate_img
    dilate_img <- dilate(plaque > 0 , kern = makeBrush(round(i*10/resolution_), shape = "disc"))
    if(region != "all") dilate_img <- dilate_img * region_img
    if(i != 1) dilate_img2 <- dilate_img * (1 - dilate_img_bk) else dilate_img2 <- dilate_img
    size_vec[i] <- sum(dilate_img2)
    dilate_mat <- as.matrix(dilate_img2@.Data)

    cell_meta[["interval"]] <- map(1:nrow(cell_meta),
                                   \(x){
                                     if(dilate_mat[cell_meta$x[x],cell_meta$y[x]] != 0 ) return(paste(i,"0um",sep = ""))
                                     else return(cell_meta[["interval"]][[x]])
                                   }) %>% unlist()
    rm(dilate_mat)

    dilate_img2 <- dilate_img2 * voronoi_img
    dilate_ls[[paste0(i,"0um_dilate_info")]] <- as.data.frame(computeFeatures.shape(dilate_img2)) %>%
      data.frame(plaque_id = as.numeric(row.names(.)), .)
  }
  gc()

  # Calc center dist
  for(i in 1:dim(plaque_meta)[1]){
    tmp_list <- apply(cell_meta, MARGIN = 1,FUN = function(x){
      tmp_dist <- sqrt((as.numeric(x[["x"]]) - plaque_meta$m.cx[i])^2+(as.numeric(x[["y"]]) - plaque_meta$m.cy[i])^2)
      if(tmp_dist < as.numeric(x[["min_center_dist"]])){
        return(list(min_dist = tmp_dist, nearest_plaque = plaque_meta$plaque_id[i]))
      }else{
        return(list(min_dist = as.numeric(x[["min_center_dist"]]), nearest_plaque = as.numeric(x[["nearest_plaque"]])))
      }
    })
    #Update cell_meta
    tmp_list <- data.frame(matrix(unlist(tmp_list), nrow = length(tmp_list), byrow = T))
    cell_meta$min_center_dist <- as.numeric(tmp_list[,1])
    cell_meta$nearest_plaque <- as.numeric(tmp_list[,2])
  }

  cell_meta$min_center_dist <- apply(cell_meta,1,
                                     FUN = function(x){
                                       plaque_num <- which(plaque_meta$plaque_id == as.numeric(x[["nearest_plaque"]]))
                                       tmp_dist <- sqrt((as.numeric(x[["x"]]) - plaque_meta$m.cx[plaque_num])^2+(as.numeric(x[["y"]]) - plaque_meta$m.cy[plaque_num])^2)
                                     })

  #Get Border Distance
  message(paste(Sys.time(),"Calc Min Dist to Plaque Border"))
  plaque_mat <- as.matrix(plaque@.Data)
  cell_meta$min_border_dist <- map(1:nrow(cell_meta),
                                   \(idx){
                                     cell_posX <- cell_meta$x[idx]
                                     cell_posY <- cell_meta$y[idx]
                                     min_center_dist <- cell_meta$min_center_dist[idx]
                                     nearest_plaque <- cell_meta$nearest_plaque[idx]
                                     j = which(plaque_meta$plaque_id == nearest_plaque)
                                     curr_dist <- max(0,min_center_dist - plaque_meta$s.radius.max[j])
                                     #Move away from border
                                     ratioX <- (cell_posX - plaque_meta$m.cx[j])/min_center_dist
                                     ratioY <- (cell_posY - plaque_meta$m.cy[j])/min_center_dist

                                     posX <- round(cell_posX - curr_dist * ratioX)
                                     posY <- round(cell_posY - curr_dist * ratioY)
                                     cycle_count <- 0
                                     while(plaque_mat[posX, posY] != nearest_plaque){
                                       curr_dist <- curr_dist + 1
                                       cycle_count <- cycle_count + 1
                                       posX <- round(cell_posX - curr_dist * ratioX)
                                       posY <- round(cell_posY - curr_dist * ratioY)
                                       if(cycle_count >= 1000){
                                         curr_dist <- -1
                                         break
                                       }
                                     }
                                     if(curr_dist != -1){
                                       curr_dist <- sqrt((posX - cell_posX)^2 + (posY - cell_posY)^2)#update actual dist
                                     }
                                     curr_dist
                                   }, .progress = T)
  rm(plaque_mat)
  gc()


  return_list <- list()
  return_list[["plaque_dilate_area"]] <- size_vec
  return_list[["cell_meta"]] <- cell_meta
  return_list[["plaque_meta"]] <- plaque_meta
  return_list[["plaque_img"]] <- plaque
  return_list[["plaque_dilate_info"]] <- dilate_ls

  starmap_obj@misc[[paste("plaque_quant",region,sep = "_")]] <- return_list

  toc <- Sys.time()
  message(paste(toc, "Done! Elapsed Time:", format(toc-tic)))
  starmap_obj
}
