#a config

a = read.csv('./2018_ml/a.csv');

levels(a$type)[levels(a$type)=="CASH_IN"] <- 1
levels(a$type)[levels(a$type)=="CASH_OUT"] <- 2
levels(a$type)[levels(a$type)=="DEBIT"] <- 3
levels(a$type)[levels(a$type)=="PAYMENT"] <- 4
levels(a$type)[levels(a$type)=="TRANSFER"] <- 5

install.packages('caret', dependencies = TRUE)
library(caret)

data(a)

smp_size = floor(0.75 * nrow(a))

set.seed(123)
train_ind = sample(seq_len(nrow(a)), size = smp_size);

train = a[train_ind, ];
test = a[-train_ind, ];

train_data = train[c(1:9)];
train_label = train$isFraud;

test_data = test[c(1:9)];
test_label = test$isFraud;

write.csv(train, file = "train_origin.csv");
