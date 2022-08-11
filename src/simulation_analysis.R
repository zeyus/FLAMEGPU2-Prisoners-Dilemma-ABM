library(tidyverse)
library(jsonlite)

# AGENT_STRATEGY_COOP 0
# AGENT_STRATEGY_DEFECT 1
# AGENT_STRATEGY_TIT_FOR_TAT 2
# AGENT_STRATEGY_RANDOM 3
strategy_ids <- c(0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33)
#strategy names
strategy_names <- c("Co-op", "Defect", "Tit-for-tat", "Random")
strategy_types <- c("Pure", "Contingent")

strategy_col_names <- c(
  "Co-op pure",
  "Co-op contingent (defect)",
  "Co-op contingent (tit-for-tat)",
  "Co-op contingent (random)",
  "Defect contingent (co-op)",
  "Defect pure",
  "Defect contingent (tit-for-tat)",
  "Defect contingent (random)",
  "Tit-for-tat contingent (co-op)",
  "Tit-for-tat contingent (defect)",
  "Tit-for-tat pure",
  "Tit-for-tat contingent (random)",
  "Random contingent (co-op)",
  "Random contingent (defect)",
  "Random contingent (tit-for-tat)",
  "Random pure"
)
#later, automate
strategy_col_map <- c(
  NA,
  "Co-op pure",
  "Co-op contingent (defect)",
  "Co-op contingent (tit-for-tat)",
  "Co-op contingent (random)",
  "Defect contingent (co-op)",
  "Defect pure",
  "Defect contingent (tit-for-tat)",
  "Defect contingent (random)",
  "Tit-for-tat contingent (co-op)",
  "Tit-for-tat contingent (defect)",
  "Tit-for-tat pure",
  "Tit-for-tat contingent (random)",
  "Random contingent (co-op)",
  "Random contingent (defect)",
  "Random contingent (tit-for-tat)",
  "Random pure",
  NA
)

strategy_col_names_df <- c(
  "step_index",
  "Co-op pure",
  "Co-op contingent (defect)",
  "Co-op contingent (tit-for-tat)",
  "Co-op contingent (random)",
  "Defect contingent (co-op)",
  "Defect pure",
  "Defect contingent (tit-for-tat)",
  "Defect contingent (random)",
  "Tit-for-tat contingent (co-op)",
  "Tit-for-tat contingent (defect)",
  "Tit-for-tat pure",
  "Tit-for-tat contingent (random)",
  "Random contingent (co-op)",
  "Random contingent (defect)",
  "Random contingent (tit-for-tat)",
  "Random pure",
  "random_seed",
  "pure_strategy",
  "cost_of_living",
  "travel_cost"
)

strategy_other_defect <- c(
  "Defect pure",
  "Co-op contingent (defect)",
  "Tit-for-tat contingent (defect)",
  "Random contingent (defect)"
)

strategy_other_coop <- c(
  "Co-op pure",
  "Defect contingent (co-op)",
  "Tit-for-tat contingent (co-op)",
  "Random contingent (co-op)"
)

strategy_other_tit_for_tat <- c(
  "Tit-for-tat pure",
  "Co-op contingent (tit-for-tat)",
  "Defect contingent (tit-for-tat)",
  "Random contingent (tit-for-tat)"
)

strategy_other_random <- c(
  "Random pure",
  "Co-op contingent (random)",
  "Defect contingent (random)",
  "Tit-for-tat contingent (random)"
)


co_op_strategies <- c(
  "Co-op pure",
  "Co-op contingent (defect)",
  "Co-op contingent (tit-for-tat)",
  "Co-op contingent (random)"
)

defect_strategies <- c(
  "Defect contingent (co-op)",
  "Defect pure",
  "Defect contingent (tit-for-tat)",
  "Defect contingent (random)"
)

tit_for_tat_strategies <- c(
  "Tit-for-tat contingent (co-op)",
  "Tit-for-tat contingent (defect)",
  "Tit-for-tat pure",
  "Tit-for-tat contingent (random)"
)

random_strategies <- c(
  "Random contingent (co-op)",
  "Random contingent (defect)",
  "Random contingent (tit-for-tat)",
  "Random pure"
)

pure_strategies <- c(
  "Co-op pure",
  "Defect pure",
  "Tit-for-tat pure",
  "Random pure"
)

contingent_strategies <- c(
  "Co-op contingent (defect)",
  "Co-op contingent (tit-for-tat)",
  "Co-op contingent (random)",
  "Defect contingent (co-op)",
  "Defect contingent (tit-for-tat)",
  "Defect contingent (random)",
  "Tit-for-tat contingent (co-op)",
  "Tit-for-tat contingent (defect)",
  "Tit-for-tat contingent (random)",
  "Random contingent (co-op)",
  "Random contingent (defect)",
  "Random contingent (tit-for-tat)"
)



df_agent_strategy <- data.frame(matrix(ncol = 17, nrow = 0))



data_dir <- "./data"

json_files <- dir(
  path = data_dir,
  pattern = ".*\\.json$",
  recursive = TRUE,
  include.dirs = TRUE,
  full.names = TRUE
)

for (json_file in json_files) {
  file_handle <- file(description = json_file, open = "r", blocking = TRUE)
  repeat{
    json_data <- readLines(file_handle, n = 1)
    if (identical(json_data, character(0))) {
      break
    }
    json_data <- fromJSON(json_data)
    df_steps <- json_data$steps
    df_sim_config <- json_data$config
    df_step_counts <- df_steps$environment
    df_data <- df_step_counts %>%
      separate(population_strat_count, strategy_col_map, sep = "[^0-9]+")
    df_data <- cbind(
      df_steps$step_index,
      df_data, df_sim_config$random_seed,
      df_sim_config$environment$strategy_pure,
      df_sim_config$environment$cost_of_living,
      df_sim_config$environment$travel_cost)
    df_agent_strategy <- rbind(df_agent_strategy, df_data)
  }
  close(file_handle)
  rm(file_handle)
  rm(df_steps)
  rm(df_step_counts)
  rm(df_data)
  rm(json_data)
}
rm(json_file)
rm(json_files)

colnames(df_agent_strategy) <- strategy_col_names_df

df_agent_strategy_long <- df_agent_strategy %>% pivot_longer(
  all_of(strategy_col_names),
  names_to = "strategy",
  values_to = "agent_count"
)
rm(df_agent_strategy)

df_agent_strategy_long$agent_count <-
  as.integer(df_agent_strategy_long$agent_count)

df_agent_strategy_long$strategy <- as.factor(df_agent_strategy_long$strategy)

# df_agent_strategy_long %>% ggplot(
#   aes(
#     x = step_index,
#     y = agent_count,
#     color = strategy,
#     group = strategy)) +
#   stat_summary(fun = "mean", geom = "line")
# first get the last step in each simulation

df_agent_strategy_long$strategy_pure <- NA
df_agent_strategy_long[df_agent_strategy_long$strategy %in% pure_strategies,]$strategy_pure <- "pure"
df_agent_strategy_long[df_agent_strategy_long$strategy %in% contingent_strategies,]$strategy_pure <- "contingent"
df_agent_strategy_long$strategy_pure <- as.factor(df_agent_strategy_long$strategy_pure)


df_agent_strategy_long$strategy_type <- NA
df_agent_strategy_long[df_agent_strategy_long$strategy %in% co_op_strategies,]$strategy_type <- "co-op"
df_agent_strategy_long[df_agent_strategy_long$strategy %in% defect_strategies,]$strategy_type <- "defect"
df_agent_strategy_long[df_agent_strategy_long$strategy %in% tit_for_tat_strategies,]$strategy_type <- "tit-for-tat"
df_agent_strategy_long[df_agent_strategy_long$strategy %in% random_strategies,]$strategy_type <- "random"
df_agent_strategy_long$strategy_type <- as.factor(df_agent_strategy_long$strategy_type)

df_agent_strategy_long$strategy_group <- paste(df_agent_strategy_long$strategy_pure, df_agent_strategy_long$strategy_type)
df_agent_strategy_long$strategy_group <- as.factor(df_agent_strategy_long$strategy_group)


df_agent_strategy_long$strategy_type_other <- NA
df_agent_strategy_long[df_agent_strategy_long$strategy %in% strategy_other_coop,]$strategy_type_other <- "co-op"
df_agent_strategy_long[df_agent_strategy_long$strategy %in% strategy_other_defect,]$strategy_type_other <- "defect"
df_agent_strategy_long[df_agent_strategy_long$strategy %in% strategy_other_tit_for_tat,]$strategy_type_other <- "tit-for-tat"
df_agent_strategy_long[df_agent_strategy_long$strategy %in% strategy_other_random,]$strategy_type_other <- "random"
df_agent_strategy_long$strategy_type_other <- as.factor(df_agent_strategy_long$strategy_type_other)

df_agent_strategy_long$environmental_harshness <- as.factor(
  round(df_agent_strategy_long$cost_of_living +
  df_agent_strategy_long$travel_cost, 3)
)

df_agent_strategy_long$pure_strategy <- as.logical(
  df_agent_strategy_long$pure_strategy)


df_agent_strategy_long_summary <- df_agent_strategy_long %>%
  group_by(random_seed, cost_of_living) %>%
  slice_max(n = 1, step_index) %>%
  ungroup()

head(df_agent_strategy_long_summary)
df_counts_mean_sd_by_strat <- df_agent_strategy_long_summary %>%
  group_by(pure_strategy, cost_of_living, strategy) %>%
  summarise(
    n = n(),
    agent_count_mean = mean(agent_count),
    agent_count_sd = sd(agent_count),
    pure_strategy = first(pure_strategy),
    cost_of_living = first(cost_of_living),
    strategy = first(strategy),
    .groups = "drop"
)

selected_env_h <- c(
  0.0000000,
  0.1000000,
  0.6666667,
  1.0000000,
  1.6660000
)

selected_env_h + selected_env_h / 2

df_counts_mean_sd_by_strat <- df_counts_mean_sd_by_strat[
  round(df_counts_mean_sd_by_strat$cost_of_living,2) %in%
    round(selected_env_h,2),
]
df_counts_mean_sd_by_strat %>% group_by(pure_strategy, cost_of_living, n) %>%
  arrange(desc(n)) %>%
  summarise(pure_strategy = first(pure_strategy),
  cost_of_living = first(cost_of_living), n = first(n))

pure_model_summary <- df_counts_mean_sd_by_strat[
  df_counts_mean_sd_by_strat$pure_strategy == TRUE &
  df_counts_mean_sd_by_strat$strategy %in% pure_strategies,]


contingent_model_summary <- df_counts_mean_sd_by_strat[
  df_counts_mean_sd_by_strat$pure_strategy == FALSE,]
contingent_model_summary  %>% group_by(pure_strategy, cost_of_living, n) %>%
  arrange(desc(n)) %>%
  summarise(pure_strategy = first(pure_strategy),
  cost_of_living = first(cost_of_living), n = first(n))

contingent_model_summary$bullshit <- NA
contingent_model_summary$bullshit <- paste0(
    round(contingent_model_summary$agent_count_mean * 10**-4, 2),
    "+/-",
    round(contingent_model_summary$agent_count_sd * 10**-4, 2))
contingent_model_summary
contingent_model_summary %>% 
  arrange(cost_of_living, strategy) %>%
  select(cost_of_living, strategy, bullshit) %>%
  reshape2::dcast(formula=strategy ~ cost_of_living, value.var="bullshit")



  reshape(idvar = "strategy", timevar = "cost_of_living", direction = "wide", v.names = c("bullshit"))



pure_model_summary %>% mutate(
  agent_count_mean = round(agent_count_mean * 10**-4, 2),
  agent_count_sd = round(agent_count_sd * 10**-4, 2)) %>%
  arrange(cost_of_living, strategy)


unique(df_counts_mean_sd_by_strat[
  df_counts_mean_sd_by_strat$pure_strategy == TRUE,]$cost_of_living)

unique(df_counts_mean_sd_by_strat[
  df_counts_mean_sd_by_strat$pure_strategy == TRUE,]$cost_of_living)

  df_counts_mean_sd_by_strat[
  df_counts_mean_sd_by_strat$pure_strategy == FALSE,]


df_agent_strategy_long_summary <-
  arrange(df_agent_strategy_long_summary, cost_of_living)

df_agent_strategy_long %>%
  filter(pure_strategy == TRUE) %>%
  ggplot(aes(x = step_index, y = agent_count, color = strategy_type, group = strategy_type)) +
  stat_summary(fun = "mean", geom = "line") +
  geom_point(size = 2) +
  facet_wrap(~cost_of_living, nrow = 2)

df_agent_strategy_long %>%
  filter(pure_strategy == FALSE) %>%
  ggplot(aes(x = step_index, y = agent_count, color = strategy_type, group = strategy_type)) +
  stat_summary(fun = "mean", geom = "line") +
  geom_point(size = 2) +
  facet_wrap(~cost_of_living, nrow = 2)

# df_agent_strategy_long$strategy_pure <- NA
# df_agent_strategy_long[df_agent_strategy_long_summary$strategy %in% pure_strategies,]$strategy_pure <- "pure"
# df_agent_strategy_long[df_agent_strategy_long_summary$strategy %in% contingent_strategies,]$strategy_pure <- "contingent"
# df_agent_strategy_long$strategy_pure <- as.factor(df_agent_strategy_long_summary$strategy_pure)


# df_agent_strategy_long$strategy_type <- NA
# df_agent_strategy_long[df_agent_strategy_long_summary$strategy %in% co_op_strategies,]$strategy_type <- "co-op"
# df_agent_strategy_long[df_agent_strategy_long_summary$strategy %in% defect_strategies,]$strategy_type <- "defect"
# df_agent_strategy_long[df_agent_strategy_long_summary$strategy %in% tit_for_tat_strategies,]$strategy_type <- "tit-for-tat"
# df_agent_strategy_long[df_agent_strategy_long_summary$strategy %in% random_strategies,]$strategy_type <- "random"
# df_agent_strategy_long$strategy_type <- as.factor(df_agent_strategy_long_summary$strategy_type)

# df_agent_strategy_long$strategy_group <- paste(df_agent_strategy_long_summary$strategy_pure, df_agent_strategy_long_summary$strategy_type)
# df_agent_strategy_long$strategy_group <- as.factor(df_agent_strategy_long_summary$stategy_group)

head(df_agent_strategy_long_summary)
unique(df_agent_strategy_long_summary$environmental_harshness)
max(df_agent_strategy_long_summary$cost_of_living)


# Pure strategy, different environmental 
df_agent_strategy_long_summary %>%
  filter(pure_strategy == TRUE & agent_count > 0) %>%
  ggplot(
    aes(
      x = environmental_harshness,
      y = agent_count,
      color = strategy_group,
      group = strategy_group)) +
  stat_summary(fun = "mean", geom = "line") +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5)


df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & agent_count > 0) %>%
  ggplot(
    aes(
      x = environmental_harshness,
      y = agent_count,
      color = strategy,
      group = strategy)) +
  stat_summary(fun = "mean", geom = "line") +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5)


df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & agent_count > 0) %>%
  ggplot(
    aes(
      x = environmental_harshness,
      y = agent_count,
      color = strategy_type,
      group = strategy_type)) +
  stat_summary(fun = "mean", geom = "line") +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5, position = "dodge")


view(df_agent_strategy_long_summary)

df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>%
  ggplot(
    aes(
      x = environmental_harshness,
      y = agent_count,
      color = strategy_type_other,
      group = strategy_type_other)) +
  stat_summary(fun = "mean", geom = "line") +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5, position = "dodge")

# Copy of above, but changed
df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>%
  ggplot(
    aes(
      x = environmental_harshness,
      y = agent_count,
      color = strategy_type_other,
      group = strategy_type_other,
      fill = strategy_type_other )) +
  geom_col(position = "dodge")
  geom_bar(position = "dodge")
  
  df_agent_strategy_long_summary %>%
    filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>%
    ggplot(
      aes(
        x = environmental_harshness, 
        y = agent_count,
        color = strategy_type_other,
        group = strategy_type_other,
        fill = strategy_type_other )) +
    geom_col(position = "dodge")
  geom_bar(position = "dodge")
  
  
  df_agent_strategy_long_summary %>%
    filter(pure_strategy == FALSE & environmental_harshness ==0) %>% 
    group_by(strategy_type_other) %>% 
    summarise(n())
  
  
## Proportional strategies, pure strategy
  
  fisken = df_agent_strategy_long_summary %>%
    filter(pure_strategy == TRUE & agent_count > 0) %>% 
    filter(environmental_harshness==0)
  
  
  fiskenC = fisken %>% filter(strategy_type_other=="co-op")
  fiskenD = fisken %>% filter(strategy_type_other=="defect")
  fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
  fiskenR = fisken %>% filter(strategy_type_other=="random")
  
  sum(fiskenC$agent_count)/sum(fisken$agent_count)
  sum(fiskenD$agent_count)/sum(fisken$agent_count)
  sum(fiskenT$agent_count)/sum(fisken$agent_count)
  sum(fiskenR$agent_count)/sum(fisken$agent_count)
  
  env_0 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
            sum(fiskenD$agent_count)/sum(fisken$agent_count),
            sum(fiskenT$agent_count)/sum(fisken$agent_count),
            sum(fiskenR$agent_count)/sum(fisken$agent_count))
  
  ##
  
  fisken = df_agent_strategy_long_summary %>%
    filter(pure_strategy == TRUE & agent_count > 0) %>% 
    filter(environmental_harshness==0.15)
  
  fiskenC = fisken %>% filter(strategy_type_other=="co-op")
  fiskenD = fisken %>% filter(strategy_type_other=="defect")
  fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
  fiskenR = fisken %>% filter(strategy_type_other=="random")
  
  sum(fiskenC$agent_count)/sum(fisken$agent_count)
  sum(fiskenD$agent_count)/sum(fisken$agent_count)
  sum(fiskenT$agent_count)/sum(fisken$agent_count)
  sum(fiskenR$agent_count)/sum(fisken$agent_count)
  
  env_0_15 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
               sum(fiskenD$agent_count)/sum(fisken$agent_count),
               sum(fiskenT$agent_count)/sum(fisken$agent_count),
               sum(fiskenR$agent_count)/sum(fisken$agent_count))
  
  ##
  fisken = df_agent_strategy_long_summary %>%
    filter(pure_strategy == TRUE & agent_count > 0) %>% 
    filter(environmental_harshness==0.45)
  
  fiskenC = fisken %>% filter(strategy_type_other=="co-op")
  fiskenD = fisken %>% filter(strategy_type_other=="defect")
  fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
  fiskenR = fisken %>% filter(strategy_type_other=="random")
  
  sum(fiskenC$agent_count)/sum(fisken$agent_count)
  sum(fiskenD$agent_count)/sum(fisken$agent_count)
  sum(fiskenT$agent_count)/sum(fisken$agent_count)
  sum(fiskenR$agent_count)/sum(fisken$agent_count)
  
  env_0_45 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
               sum(fiskenD$agent_count)/sum(fisken$agent_count),
               sum(fiskenT$agent_count)/sum(fisken$agent_count),
               sum(fiskenR$agent_count)/sum(fisken$agent_count))
  
  ##
  
  fisken = df_agent_strategy_long_summary %>%
    filter(pure_strategy == TRUE & agent_count > 0) %>% 
    filter(environmental_harshness==1)
  
  fiskenC = fisken %>% filter(strategy_type_other=="co-op")
  fiskenD = fisken %>% filter(strategy_type_other=="defect")
  fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
  fiskenR = fisken %>% filter(strategy_type_other=="random")
  
  sum(fiskenC$agent_count)/sum(fisken$agent_count)
  sum(fiskenD$agent_count)/sum(fisken$agent_count)
  sum(fiskenT$agent_count)/sum(fisken$agent_count)
  sum(fiskenR$agent_count)/sum(fisken$agent_count)
  
  env_1 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
            sum(fiskenD$agent_count)/sum(fisken$agent_count),
            sum(fiskenT$agent_count)/sum(fisken$agent_count),
            sum(fiskenR$agent_count)/sum(fisken$agent_count))
  
  
  ##
  
  fisken = df_agent_strategy_long_summary %>%
    filter(pure_strategy == TRUE & agent_count > 0) %>% 
    filter(environmental_harshness==1.5)
  
  fiskenC = fisken %>% filter(strategy_type_other=="co-op")
  fiskenD = fisken %>% filter(strategy_type_other=="defect")
  fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
  fiskenR = fisken %>% filter(strategy_type_other=="random")
  
  sum(fiskenC$agent_count)/sum(fisken$agent_count)
  sum(fiskenD$agent_count)/sum(fisken$agent_count)
  sum(fiskenT$agent_count)/sum(fisken$agent_count)
  sum(fiskenR$agent_count)/sum(fisken$agent_count)
  
  env_1_5 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
              sum(fiskenD$agent_count)/sum(fisken$agent_count),
              sum(fiskenT$agent_count)/sum(fisken$agent_count),
              sum(fiskenR$agent_count)/sum(fisken$agent_count))
  
  ##
  
  fisken = df_agent_strategy_long_summary %>%
    filter(pure_strategy == TRUE & agent_count > 0) %>% 
    filter(environmental_harshness==2)
  
  fiskenC = fisken %>% filter(strategy_type_other=="co-op")
  fiskenD = fisken %>% filter(strategy_type_other=="defect")
  fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
  fiskenR = fisken %>% filter(strategy_type_other=="random")
  
  sum(fiskenC$agent_count)/sum(fisken$agent_count)
  sum(fiskenD$agent_count)/sum(fisken$agent_count)
  sum(fiskenT$agent_count)/sum(fisken$agent_count)
  sum(fiskenR$agent_count)/sum(fisken$agent_count)
  
  env_2 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
            sum(fiskenD$agent_count)/sum(fisken$agent_count),
            sum(fiskenT$agent_count)/sum(fisken$agent_count),
            sum(fiskenR$agent_count)/sum(fisken$agent_count))
  
  ##
  
  
  prop_0 = cbind(env_0,rep("0",4),c("co-op","defect","tit-for-tat","random"))
  prop_0_15 = cbind(env_0_15,rep("0.15",4),c("co-op","defect","tit-for-tat","random"))
  prop_0_45 = cbind(env_0_45,rep("0.45",4),c("co-op","defect","tit-for-tat","random"))
  prop_1 = cbind(env_1,rep("1",4),c("co-op","defect","tit-for-tat","random"))
  prop_1_5 = cbind(env_1_5,rep("1.5",4),c("co-op","defect","tit-for-tat","random"))
  prop_2 = cbind(env_2,rep("2",4),c("co-op","defect","tit-for-tat","random"))
  # prop_3 = cbind(env_3,rep("3",4),c("co-op","defect","tit-for-tat","random"))
  
  skrrt_prop = as.data.frame(rbind(prop_0,prop_0_15, prop_0_45, prop_1, prop_1_5,prop_2))
  
  skrrt_prop[,1] = as.numeric(skrrt_prop[,1])
  
  skrrt_prop = rename(skrrt_prop, Strategy_type = V3, Environment_cost = V2,Proportion_strategy = env_0)
  
  skrrt_prop %>%
    ggplot(
      aes(
        x = Environment_cost,
        y = Proportion_strategy,
        color = Strategy_type,
        group = Strategy_type,
        fill = Strategy_type )) +
    geom_col(position = "dodge")
  
  
  
  
  
  
  ## Making proportional graphs of the strategies
  
  ## Start for strategy type
  
 fisken = df_agent_strategy_long_summary %>%
    filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
   filter(environmental_harshness==0)
 
 fiskenC = fisken %>% filter(strategy_type=="co-op")
 fiskenD = fisken %>% filter(strategy_type=="defect")
 fiskenT = fisken %>% filter(strategy_type=="tit-for-tat")
 fiskenR = fisken %>% filter(strategy_type=="random")
 
 sum(fiskenC$agent_count)/sum(fisken$agent_count)
 sum(fiskenD$agent_count)/sum(fisken$agent_count)
 sum(fiskenT$agent_count)/sum(fisken$agent_count)
 sum(fiskenR$agent_count)/sum(fisken$agent_count)
 
 env_0 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
    sum(fiskenD$agent_count)/sum(fisken$agent_count),
    sum(fiskenT$agent_count)/sum(fisken$agent_count),
    sum(fiskenR$agent_count)/sum(fisken$agent_count))
 
 ##
 
fisken = df_agent_strategy_long_summary %>%
   filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
   filter(environmental_harshness==0.15)
 
 fiskenC = fisken %>% filter(strategy_type=="co-op")
 fiskenD = fisken %>% filter(strategy_type=="defect")
 fiskenT = fisken %>% filter(strategy_type=="tit-for-tat")
 fiskenR = fisken %>% filter(strategy_type=="random")
 
 sum(fiskenC$agent_count)/sum(fisken$agent_count)
 sum(fiskenD$agent_count)/sum(fisken$agent_count)
 sum(fiskenT$agent_count)/sum(fisken$agent_count)
 sum(fiskenR$agent_count)/sum(fisken$agent_count)
 
 env_0_15 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
           sum(fiskenD$agent_count)/sum(fisken$agent_count),
           sum(fiskenT$agent_count)/sum(fisken$agent_count),
           sum(fiskenR$agent_count)/sum(fisken$agent_count))
 
 ##
 fisken = df_agent_strategy_long_summary %>%
   filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
   filter(environmental_harshness==0.45)
 
 fiskenC = fisken %>% filter(strategy_type=="co-op")
 fiskenD = fisken %>% filter(strategy_type=="defect")
 fiskenT = fisken %>% filter(strategy_type=="tit-for-tat")
 fiskenR = fisken %>% filter(strategy_type=="random")
 
 sum(fiskenC$agent_count)/sum(fisken$agent_count)
 sum(fiskenD$agent_count)/sum(fisken$agent_count)
 sum(fiskenT$agent_count)/sum(fisken$agent_count)
 sum(fiskenR$agent_count)/sum(fisken$agent_count)
 
 env_0_45 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
              sum(fiskenD$agent_count)/sum(fisken$agent_count),
              sum(fiskenT$agent_count)/sum(fisken$agent_count),
              sum(fiskenR$agent_count)/sum(fisken$agent_count))
 
##
 
 fisken = df_agent_strategy_long_summary %>%
   filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
   filter(environmental_harshness==1)
 
 fiskenC = fisken %>% filter(strategy_type=="co-op")
 fiskenD = fisken %>% filter(strategy_type=="defect")
 fiskenT = fisken %>% filter(strategy_type=="tit-for-tat")
 fiskenR = fisken %>% filter(strategy_type=="random")
 
 sum(fiskenC$agent_count)/sum(fisken$agent_count)
 sum(fiskenD$agent_count)/sum(fisken$agent_count)
 sum(fiskenT$agent_count)/sum(fisken$agent_count)
 sum(fiskenR$agent_count)/sum(fisken$agent_count)
 
 env_1 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
              sum(fiskenD$agent_count)/sum(fisken$agent_count),
              sum(fiskenT$agent_count)/sum(fisken$agent_count),
              sum(fiskenR$agent_count)/sum(fisken$agent_count))
 
 
 ##
 
 fisken = df_agent_strategy_long_summary %>%
   filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
   filter(environmental_harshness==1.5)
 
 fiskenC = fisken %>% filter(strategy_type=="co-op")
 fiskenD = fisken %>% filter(strategy_type=="defect")
 fiskenT = fisken %>% filter(strategy_type=="tit-for-tat")
 fiskenR = fisken %>% filter(strategy_type=="random")
 
 sum(fiskenC$agent_count)/sum(fisken$agent_count)
 sum(fiskenD$agent_count)/sum(fisken$agent_count)
 sum(fiskenT$agent_count)/sum(fisken$agent_count)
 sum(fiskenR$agent_count)/sum(fisken$agent_count)
 
 env_1_5 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
           sum(fiskenD$agent_count)/sum(fisken$agent_count),
           sum(fiskenT$agent_count)/sum(fisken$agent_count),
           sum(fiskenR$agent_count)/sum(fisken$agent_count))
 
 ##
 
 fisken = df_agent_strategy_long_summary %>%
   filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
   filter(environmental_harshness==2)
 
 fiskenC = fisken %>% filter(strategy_type=="co-op")
 fiskenD = fisken %>% filter(strategy_type=="defect")
 fiskenT = fisken %>% filter(strategy_type=="tit-for-tat")
 fiskenR = fisken %>% filter(strategy_type=="random")
 
 sum(fiskenC$agent_count)/sum(fisken$agent_count)
 sum(fiskenD$agent_count)/sum(fisken$agent_count)
 sum(fiskenT$agent_count)/sum(fisken$agent_count)
 sum(fiskenR$agent_count)/sum(fisken$agent_count)
 
 env_2 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
             sum(fiskenD$agent_count)/sum(fisken$agent_count),
             sum(fiskenT$agent_count)/sum(fisken$agent_count),
             sum(fiskenR$agent_count)/sum(fisken$agent_count))
 
 ##

 
 prop_0 = cbind(env_0,rep("0",4),c("co-op","defect","tit-for-tat","random"))
 prop_0_15 = cbind(env_0_15,rep("0.15",4),c("co-op","defect","tit-for-tat","random"))
 prop_0_45 = cbind(env_0_45,rep("0.45",4),c("co-op","defect","tit-for-tat","random"))
 prop_1 = cbind(env_1,rep("1",4),c("co-op","defect","tit-for-tat","random"))
 prop_1_5 = cbind(env_1_5,rep("1.5",4),c("co-op","defect","tit-for-tat","random"))
 prop_2 = cbind(env_2,rep("2",4),c("co-op","defect","tit-for-tat","random"))
 # prop_3 = cbind(env_3,rep("3",4),c("co-op","defect","tit-for-tat","random"))
 
 skrrt_prop = as.data.frame(rbind(prop_0,prop_0_15, prop_0_45, prop_1, prop_1_5,prop_2))
 
 skrrt_prop[,1] = as.numeric(skrrt_prop[,1])
 
 skrrt_prop = rename(skrrt_prop, Strategy_type = V3, Environment_cost = V2,Proportion_strategy = env_0)
 
skrrt_prop %>%
   ggplot(
     aes(
       x = Environment_cost,
       y = Proportion_strategy,
       color = Strategy_type,
       group = Strategy_type,
       fill = Strategy_type )) +
   geom_col(position = "dodge")





## Start Strategy_type_other

fisken = df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
  filter(environmental_harshness==0)


fiskenC = fisken %>% filter(strategy_type_other=="co-op")
fiskenD = fisken %>% filter(strategy_type_other=="defect")
fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
fiskenR = fisken %>% filter(strategy_type_other=="random")

sum(fiskenC$agent_count)/sum(fisken$agent_count)
sum(fiskenD$agent_count)/sum(fisken$agent_count)
sum(fiskenT$agent_count)/sum(fisken$agent_count)
sum(fiskenR$agent_count)/sum(fisken$agent_count)

env_0 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
          sum(fiskenD$agent_count)/sum(fisken$agent_count),
          sum(fiskenT$agent_count)/sum(fisken$agent_count),
          sum(fiskenR$agent_count)/sum(fisken$agent_count))

##

fisken = df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
  filter(environmental_harshness==0.15)

fiskenC = fisken %>% filter(strategy_type_other=="co-op")
fiskenD = fisken %>% filter(strategy_type_other=="defect")
fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
fiskenR = fisken %>% filter(strategy_type_other=="random")

sum(fiskenC$agent_count)/sum(fisken$agent_count)
sum(fiskenD$agent_count)/sum(fisken$agent_count)
sum(fiskenT$agent_count)/sum(fisken$agent_count)
sum(fiskenR$agent_count)/sum(fisken$agent_count)

env_0_15 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
             sum(fiskenD$agent_count)/sum(fisken$agent_count),
             sum(fiskenT$agent_count)/sum(fisken$agent_count),
             sum(fiskenR$agent_count)/sum(fisken$agent_count))

##
fisken = df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
  filter(environmental_harshness==0.45)

fiskenC = fisken %>% filter(strategy_type_other=="co-op")
fiskenD = fisken %>% filter(strategy_type_other=="defect")
fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
fiskenR = fisken %>% filter(strategy_type_other=="random")

sum(fiskenC$agent_count)/sum(fisken$agent_count)
sum(fiskenD$agent_count)/sum(fisken$agent_count)
sum(fiskenT$agent_count)/sum(fisken$agent_count)
sum(fiskenR$agent_count)/sum(fisken$agent_count)

env_0_45 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
             sum(fiskenD$agent_count)/sum(fisken$agent_count),
             sum(fiskenT$agent_count)/sum(fisken$agent_count),
             sum(fiskenR$agent_count)/sum(fisken$agent_count))

##

fisken = df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
  filter(environmental_harshness==1)

fiskenC = fisken %>% filter(strategy_type_other=="co-op")
fiskenD = fisken %>% filter(strategy_type_other=="defect")
fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
fiskenR = fisken %>% filter(strategy_type_other=="random")

sum(fiskenC$agent_count)/sum(fisken$agent_count)
sum(fiskenD$agent_count)/sum(fisken$agent_count)
sum(fiskenT$agent_count)/sum(fisken$agent_count)
sum(fiskenR$agent_count)/sum(fisken$agent_count)

env_1 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
          sum(fiskenD$agent_count)/sum(fisken$agent_count),
          sum(fiskenT$agent_count)/sum(fisken$agent_count),
          sum(fiskenR$agent_count)/sum(fisken$agent_count))


##

fisken = df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
  filter(environmental_harshness==1.5)

fiskenC = fisken %>% filter(strategy_type_other=="co-op")
fiskenD = fisken %>% filter(strategy_type_other=="defect")
fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
fiskenR = fisken %>% filter(strategy_type_other=="random")

sum(fiskenC$agent_count)/sum(fisken$agent_count)
sum(fiskenD$agent_count)/sum(fisken$agent_count)
sum(fiskenT$agent_count)/sum(fisken$agent_count)
sum(fiskenR$agent_count)/sum(fisken$agent_count)

env_1_5 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
            sum(fiskenD$agent_count)/sum(fisken$agent_count),
            sum(fiskenT$agent_count)/sum(fisken$agent_count),
            sum(fiskenR$agent_count)/sum(fisken$agent_count))

##

fisken = df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & environmental_harshness !=7.5) %>% 
  filter(environmental_harshness==2)

fiskenC = fisken %>% filter(strategy_type_other=="co-op")
fiskenD = fisken %>% filter(strategy_type_other=="defect")
fiskenT = fisken %>% filter(strategy_type_other=="tit-for-tat")
fiskenR = fisken %>% filter(strategy_type_other=="random")

sum(fiskenC$agent_count)/sum(fisken$agent_count)
sum(fiskenD$agent_count)/sum(fisken$agent_count)
sum(fiskenT$agent_count)/sum(fisken$agent_count)
sum(fiskenR$agent_count)/sum(fisken$agent_count)

env_2 = c(sum(fiskenC$agent_count)/sum(fisken$agent_count),
          sum(fiskenD$agent_count)/sum(fisken$agent_count),
          sum(fiskenT$agent_count)/sum(fisken$agent_count),
          sum(fiskenR$agent_count)/sum(fisken$agent_count))

##


prop_0 = cbind(env_0,rep("0",4),c("co-op","defect","tit-for-tat","random"))
prop_0_15 = cbind(env_0_15,rep("0.15",4),c("co-op","defect","tit-for-tat","random"))
prop_0_45 = cbind(env_0_45,rep("0.45",4),c("co-op","defect","tit-for-tat","random"))
prop_1 = cbind(env_1,rep("1",4),c("co-op","defect","tit-for-tat","random"))
prop_1_5 = cbind(env_1_5,rep("1.5",4),c("co-op","defect","tit-for-tat","random"))
prop_2 = cbind(env_2,rep("2",4),c("co-op","defect","tit-for-tat","random"))
# prop_3 = cbind(env_3,rep("3",4),c("co-op","defect","tit-for-tat","random"))

skrrt_prop = as.data.frame(rbind(prop_0,prop_0_15, prop_0_45, prop_1, prop_1_5,prop_2))

skrrt_prop[,1] = as.numeric(skrrt_prop[,1])

skrrt_prop = rename(skrrt_prop, Strategy_type = V3, Environment_cost = V2,Proportion_strategy = env_0)

skrrt_prop %>%
  ggplot(
    aes(
      x = Environment_cost,
      y = Proportion_strategy,
      color = Strategy_type,
      group = Strategy_type,
      fill = Strategy_type )) +
  geom_col(position = "dodge")






 
 
 
 
 class(fisken$agent_count)
 
 sum(fiskenC$agent_count)
 
 
 
 view(fisken)
 
 %>% 
   filter(strategy_type)
 
 
 
   
    group_by(environmental_harshness) %>% 
    group_by(strategy_type,agent_count) %>% sum(agent_count)
   
   
   count()
   
   
    summarise()
  
  
  
  
  
  view(agent_count)
  
  view(df_agent_strategy_long_summary$agent_count)
  view(df_agent_strategy_long)
  
df_agent_strategy_long_summary
  
  
  
  
  stat_summary(fun = "mean", geom = "line") +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5)





df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & agent_count > 0) %>%
  ggplot(
    aes(
      x = environmental_harshness,
      y = agent_count,
      color = strategy_type_other,
      group = strategy_type_other)) +
  stat_summary(fun = "mean", geom = "line") +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5)

df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & agent_count > 0) %>%
  ggplot(
    aes(
      x = environmental_harshness,
      y = agent_count,
      color = strategy_type,
      group = strategy_type)) +
  stat_summary(fun = "mean", geom = "line") +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5)

df_agent_strategy_long_summary %>%
  ggplot(
    aes(
      x = environmental_harshness,
      y = agent_count,
      color = strategy_group,
      group = strategy_group)) +
  stat_summary(fun = "mean", geom = "line") +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5) +
  facet_wrap(vars(pure_strategy))


df_agent_strategy_long_summary %>% filter(
    strategy_group == "pure tit-for-tat" & environmental_harshness == 1.5)
