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
  NA,
  NA,
  NA,
  NA,
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
  "Random pure"
)

df_agent_strategy <- data.frame(matrix(ncol = 17, nrow = 0))



json_file <- "data/pure0_env_cost10_10000_steps/4.json"

# note, the JSON files append a new json object,
# so we need to read in the whole file.
file_handle <- file(description = json_file, open = "r", blocking = TRUE)
repeat{
  json_data <- readLines(file_handle, n = 1)
  if (identical(json_data, character(0))) {
    break
  }
  json_data <- fromJSON(json_data)
  df_steps <- json_data$steps
  df_step_counts <- df_steps$environment
  df_data <- df_step_counts %>%
    separate(population_counts_step, strategy_col_map)
  df_data <- cbind(df_steps$step_index, df_data)
  df_agent_strategy <- rbind(df_agent_strategy, df_data)
}
close(file_handle)
rm(file_handle)
rm(df_steps)
rm(df_step_counts)
rm(df_data)
rm(json_data)
rm(json_file)

colnames(df_agent_strategy) <- strategy_col_names_df

head(df_agent_strategy)


df_agent_strategy_long <- df_agent_strategy %>% pivot_longer(
  strategy_col_names,
  names_to = "strategy",
  values_to = "agent_count"
)

df_agent_strategy_long$agent_count <-
  as.numeric(df_agent_strategy_long$agent_count)

df_agent_strategy_long$strategy <- as.factor(df_agent_strategy_long$strategy)

head(df_agent_strategy_long, n = 100)

df_agent_strategy_long %>% ggplot(
  aes(
    x = step_index,
    y = agent_count,
    color = strategy,
    group = strategy)) +
  stat_summary(fun = "mean", geom = "line")
