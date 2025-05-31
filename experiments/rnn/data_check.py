from utils.ibdm_data_loader import IMDBDataLoader

dataloader = IMDBDataLoader()
train_data, valid_data, test_data = dataloader.load_data()

# Get a batch from the train data
if __name__ == '__main__':
    for batch in train_data:
        text, labels = batch
        decoded_text = dataloader.decode_embeddings_to_text(text[0])
        print(decoded_text)
        print("Sample batch:")
        print(f"Text shape: {text.shape}")
        print(f"Labels shape: {labels.shape}")
        print("\nFirst text sample:", text[0])
        print("First label:", labels[0])
        # break











