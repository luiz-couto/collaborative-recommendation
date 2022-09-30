import pandas as pd
import numpy as np

LEARNING_RATE = 0.012
REGULARIZATION_FACTOR = 0.01
N_FACTORS = 5
N_EPOCHS = 40

MIN_RATING = 0
MAX_RATING = 5

# Esse método realiza o mapeamento de uma coluna do dataset para seu relativo indice
def generateMapping(df, columnName):
    ids = df[columnName].unique().tolist()
    return dict(zip(ids, range(len(ids))))


# Esse método gera um novo data set com as tuplas (usuário, item, nota) com 
# os ids dos usuários e dos itens convertidos para seus respectivos índices
# em número inteiro
def generateMappedDataset(df, userMapping, itemMapping):
    copied = df.copy()
    copied['UserId'] = copied['UserId'].map(userMapping)
    copied['ItemId'] = copied['ItemId'].map(itemMapping)

    copied.fillna(-1, inplace=True)

    copied['UserId'] = copied['UserId'].astype(np.int32)
    copied['ItemId'] = copied['ItemId'].astype(np.int32)

    return copied[['UserId', 'ItemId', 'Rating']].to_numpy()


# Esse método executa uma versão da ideia do SGD (Stochastic Gradient Descent)
# para treino dos fatores latentes de usuários e itens.
def getLatentFactors(training, validation, globalMean):
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

        rmse = computeRMSE(validation, userLatents, itemsLatents, globalMean)
        print(rmse)

    return userLatents, itemsLatents


# Após o aprendizado dos fatores latentes para as matrizes de usuários e
# itens, esse método realiza a predição de notas para o dado conjunto de
# targets.
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
    ratings_df = pd.read_csv('ratings.csv', encoding='latin-1', sep=',|:', engine='python')
    targets = pd.read_csv('targets.csv', sep=':', engine='python')



    training = ratings_df[['UserId', 'ItemId', 'Rating']].sample(frac=0.8, random_state=8)
    validation = ratings_df.drop(training.index.tolist())
    
    userMapping = generateMapping(training, 'UserId')
    itemMapping = generateMapping(training, 'ItemId')

    training = generateMappedDataset(training, userMapping, itemMapping)
    validation = generateMappedDataset(validation, userMapping, itemMapping)
    
    targets = zip(targets['UserId'], targets['ItemId'])

    globalMean = np.mean(training[:, 2])

    userLatents, itemLatents = getLatentFactors(training, validation, globalMean)

    predictions = getPredicitons(targets, userMapping, itemMapping, globalMean, userLatents, itemLatents)

    # print('UserId:ItemId,Rating')
    # for user, item, pred in predictions:
    #     print(str(user) + ":" + str(item) + "," + str(pred))



def computeRMSE(validation, userLatents, itemLatents, globalMean):
    error = []

    for i in range(validation.shape[0]):
        user = int(validation[i, 0])
        item = int(validation[i, 1])
        rating = validation[i, 2]
        prediction = globalMean

        if user > -1 and item > -1:
            for factor in range(N_FACTORS):
                prediction += userLatents[user, factor] * itemLatents[item, factor]

        error.append(rating - prediction)
        
    error = np.array(error)
    rmse = np.sqrt((np.power(error,2)).mean())

    return rmse


if __name__ == "__main__":
    main()


