import sys
import pandas as pd
import numpy as np

LEARNING_RATE = 0.005
REGULARIZATION_FACTOR = 0.008
N_FACTORS = 4
N_EPOCHS = 20

MIN_RATING = 0
MAX_RATING = 5

# Maps a column of the dataset to its index
def generateMapping(df, columnName):
    ids = df[columnName].unique().tolist()
    return dict(zip(ids, range(len(ids))))


# Generate the tuples (user, item, rating)
def generateMappedDataset(df, userMapping, itemMapping):
    copied = df.copy()
    copied['UserId'] = copied['UserId'].map(userMapping)
    copied['ItemId'] = copied['ItemId'].map(itemMapping)

    copied.fillna(-1, inplace=True)

    copied['UserId'] = copied['UserId'].astype(np.int32)
    copied['ItemId'] = copied['ItemId'].astype(np.int32)

    return copied[['UserId', 'ItemId', 'Rating']].to_numpy()


# Run the SGD (Stochastic Gradient Descent) for training of the
# latent factors of users and items
def getLatentFactors(training, globalMean):
    np.random.seed(seed=13)
    userLatents = np.random.normal(0, .01, (len(np.unique(training[:, 0])), N_FACTORS))
    itemsLatents = np.random.normal(0, .01, (len(np.unique(training[:, 1])), N_FACTORS))
    
    for epoch in range(N_EPOCHS):
        for i in range(training.shape[0]):
            user = int(training[i, 0])
            item = int(training[i, 1])
            rating = training[i, 2]

            prediction = globalMean

            for factor in range(N_FACTORS):
                prediction += userLatents[user, factor] * itemsLatents[item, factor]
            
            error = rating - prediction

            for factor in range(N_FACTORS):
                userFactor = userLatents[user, factor]
                itemFactor = itemsLatents[item, factor]

                userLatents[user, factor] += LEARNING_RATE * (error * itemFactor - REGULARIZATION_FACTOR * userFactor)
                itemsLatents[item, factor] += LEARNING_RATE * (error * userFactor - REGULARIZATION_FACTOR * itemFactor)

    return userLatents, itemsLatents


# Make the predictions of the targets based on the latent factors
def getPredicitons(targets, userMapping, itemMapping, globalMean, userLatents, itemLatents):
    predictions = []

    for user, item in targets:
        pred = globalMean

        if user in userMapping and item in itemMapping:
            userIndex = userMapping[user]
            itemIndex = itemMapping[item]
            pred += np.dot(userLatents[userIndex], itemLatents[itemIndex])
        
        if pred > MAX_RATING:
            pred = MAX_RATING
        
        if pred < MIN_RATING:
            pred = MIN_RATING
        
        predictions.append([user, item, pred])

    return predictions


def main():

    if len(sys.argv) != 3:
        print("usage: python3 main.py ratings.csv targets.csv")
        exit(1)
    
    ratings_df = pd.read_csv(sys.argv[1], encoding='latin-1', sep=',|:', engine='python')
    targets = pd.read_csv(sys.argv[2], sep=':', engine='python')

    userMapping = generateMapping(ratings_df, 'UserId')
    itemMapping = generateMapping(ratings_df, 'ItemId')

    training = generateMappedDataset(ratings_df, userMapping, itemMapping)
    targets = zip(targets['UserId'], targets['ItemId'])

    globalMean = np.mean(training[:, 2])

    userLatents, itemLatents = getLatentFactors(training, globalMean)

    predictions = getPredicitons(targets, userMapping, itemMapping, globalMean, userLatents, itemLatents)

    print('UserId:ItemId,Rating')
    for user, item, pred in predictions:
        print(str(user) + ":" + str(item) + "," + str(pred))


if __name__ == "__main__":
    main()


