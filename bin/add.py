import torch, tqdm
import torch.utils.data

def create(N, batch):
    X = torch.rand(300, 2)
    Y = (X[:,0]-X[:,1]).abs()

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch)

    return dataloader

def main():

    model = torch.nn.Sequential(

        torch.nn.Linear(1, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)

    )

    epochs = 300
    N_data = 300
    N_test = 300
    
    lossf = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.99, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    with tqdm.tqdm(range(epochs)) as bar:
        for epoch in bar:

            data = create(300, batch=32)

            model.train()
            for x, y in data:
                yh = (model(x[:,:1])-model(x[:,1:])).squeeze(-1).abs()
                loss = lossf(yh, y)
                optim.zero_grad()
                loss.backward()
                optim.step()

            sched.step()

            model.eval()

            with torch.no_grad():

                test = create(300, batch=300)

                for i, (x, y) in enumerate(test):
                    yh = (model(x[:,:1])-model(x[:,1:])).squeeze(-1).abs()
                    rmse = lossf(yh, y) ** 0.5

                assert i == 0

            bar.set_description("Test %.4f" % rmse.item())
                    
        
if __name__ == "__main__":
    main()
