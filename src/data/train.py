import torch
from torch.utils.data import DataLoader
from Dataloader import ToxicDataset

def main():
    # Use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # Hyper parameters
    batch_size = 8
    num_workers = 2
    num_epochs = 10

    toxic_dataset = ToxicDataset(train_split=True)
    toxic_dataloader = DataLoader(dataset=toxic_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for epoch in range(num_epochs):
        for batch_idx, batch_sample in enumerate(toxic_dataloader):
            data, target = batch_sample
            print(data[0])
            print(target[0])
            break
        break

if __name__ == "__main__":
    main()