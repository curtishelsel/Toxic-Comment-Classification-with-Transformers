from models.transformer import Transformer
import models.train_model as train_model 

if __name__ == '__main__':
    
    model = Transformer(num_tokens=4, dim_model=8, num_heads=2, 
            num_enc_layers=3, num_dec_layers=3, dropout=0.1)


    train_model.fit(model, epochs = 13)
