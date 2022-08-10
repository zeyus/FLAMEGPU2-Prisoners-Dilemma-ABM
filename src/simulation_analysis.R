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
  strategy_col_names,
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
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5)



skrrtPlot =  df_agent_strategy_long_summary %>%
  filter(pure_strategy == FALSE & agent_count > 0) %>%
  ggplot(
    aes(
      x = environmental_harshness,
      y = agent_count,
      color = strategy,
      group = strategy)) +
  stat_summary(fun = "mean", geom = "line") +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.5);skrrtPlot




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
