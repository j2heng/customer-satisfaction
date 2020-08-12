from data.data_generator import DataGenerator
from models.base_model import BiGRUAttention, BiLSTMAttention
from models.base_trainer import SimpleTrainer, CBTrainer
from utils.utils import get_configs


def main():
    print("------- get configs -------")
    config = get_configs("./py/configs.json")

    print("------- generate dataset -------")
    data = DataGenerator(config)
    X_train, y_train = data.get_train_data()
    X_test, y_test = data.get_test_data()
    word_index = data.get_word_index()

    print("------- create model -------")
    model = BiGRUAttention(config, word_index)
    
    print("------- model training and predicting-------")
    trainer = CBTrainer(model, config)
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    trainer.score(X_test, y_test)

    return y_pred

if __name__ == '__main__':
    main()
    