import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


class LeNet5(torch.nn.Module):
    def __init__(self, activation='tanh', pooling='avg', conv_size=5, use_batch_norm=False):
        super(LeNet5, self).__init__()

        self.conv_size = conv_size
        self.use_batch_norm = use_batch_norm

        act_fn = torch.nn.Tanh() if activation == 'tanh' else torch.nn.ReLU()
        pool = torch.nn.AvgPool2d(2, 2) if pooling == 'avg' else torch.nn.MaxPool2d(2, 2)

        if conv_size == 5:
            self.conv1=torch.nn.Conv2d(1, 6, kernel_size=conv_size, padding=2)
        elif conv_size == 3:
            self.conv1_1 = torch.nn.Conv2d(1, 6, kernel_size=conv_size, padding=1)
            self.conv1_2 = torch.nn.Conv2d(6, 6, kernel_size=conv_size, padding=1)

        self.bn1= torch.nn.BatchNorm2d(6) 
        self.act1=act_fn
        self.pool1=pool

        if conv_size == 5:
            self.conv2=torch.nn.Conv2d(6, 16, kernel_size=conv_size, padding=0)
        elif conv_size == 3:
            self.conv2_1 = torch.nn.Conv2d(6, 16, kernel_size=conv_size, padding=0)
            self.conv2_2 = torch.nn.Conv2d(16, 16, kernel_size=conv_size, padding=0)

        self.bn2= torch.nn.BatchNorm2d(16) 
        self.act2=act_fn
        self.pool2=pool

        self.fc3=torch.nn.Linear(5*5*16, 120)
        self.act3=act_fn
        self.fc4=torch.nn.Linear(120, 84)
        self.act4=act_fn
        self.fc5=torch.nn.Linear(84, 10)


    def forward(self, x):

        if self.conv_size == 5:
            x = self.conv1(x)
        elif self.conv_size == 3:
            x = self.conv1_2(self.conv1_1(x))

        if self.use_batch_norm:
            x = self.bn1(x)

        x = self.act1(x)
        x = self.pool1(x)

        if self.conv_size == 5:
            x = self.conv2(x)
        elif self.conv_size == 3:
            x = self.conv2_2(self.conv2_1(x))

        if self.use_batch_norm:
            x = self.bn2(x)

        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc3(x)
        x = self.act3(x)

        x = self.fc4(x)
        x = self.act4(x)

        x = self.fc5(x)

        return x


def train_and_evaluate(system_name, X_train, y_train, X_test, y_test,
                        activation='tanh', pooling='avg', conv_size=5, use_batch_norm=False,
                        num_epochs=50, batch_size=100):
    print(f"\n--- Обучение для {system_name.replace('_', ' ').title()} | activation={activation} pooling={pooling} conv={conv_size} bn={use_batch_norm} ---")

    model = LeNet5(activation=activation, pooling=pooling, conv_size=conv_size, use_batch_norm=use_batch_norm)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    acc_hist = []
    loss_hist = []

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            x_batch = X_train[idx].to(device)
            y_batch = y_train[idx].to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_test.to(device))
            val_loss = loss_fn(val_output, y_test.to(device)).item()
            val_acc = (val_output.argmax(1) == y_test.to(device)).float().mean().item()

        acc_hist.append(val_acc)
        loss_hist.append(val_loss)

        print(f" Эпоха {epoch + 1}/{num_epochs}: Точность={val_acc:.4f} Потери={val_loss:.4f}")

    return acc_hist, loss_hist

if __name__ == "__main__":
    dataset_path = "digit_dataset"
    systems = ["western", "eastern_arabic", "roman"]
    num_epochs = 50
    plots_dir = "training_plots"
    os.makedirs(plots_dir, exist_ok=True)

    configs = [
        {"activation": "tanh", "pooling": "avg", "conv_size": 5, "use_batch_norm": False},
        {"activation": "relu", "pooling": "avg", "conv_size": 5, "use_batch_norm": False},
        {"activation": "relu", "pooling": "max", "conv_size": 3, "use_batch_norm": False},
        {"activation": "relu", "pooling": "max", "conv_size": 3, "use_batch_norm": True}
    ]

    results = {}
    for system in systems:
        try:
            X_train = torch.load(os.path.join(dataset_path, f"{system}_X_train.pt"))
            y_train = torch.load(os.path.join(dataset_path, f"{system}_y_train.pt"))
            X_test = torch.load(os.path.join(dataset_path, f"{system}_X_test.pt"))
            y_test = torch.load(os.path.join(dataset_path, f"{system}_y_test.pt"))

            for cfg in configs:
                if int(cfg['use_batch_norm']):
                    label = f"{system}_{cfg['activation']}_{cfg['pooling']}_{cfg['conv_size']}_bn"
                else:
                    label = f"{system}_{cfg['activation']}_{cfg['pooling']}_{cfg['conv_size']}"
                acc_hist, loss_hist = train_and_evaluate(system, X_train, y_train, X_test, y_test,
                                                         activation=cfg['activation'], pooling=cfg['pooling'],
                                                         conv_size=cfg['conv_size'], use_batch_norm=cfg['use_batch_norm'],
                                                         num_epochs=num_epochs)

                results[label] = {
                    "system": system,
                    "activation": cfg['activation'],
                    "pooling": cfg['pooling'],
                    "conv_size": cfg['conv_size'],
                    "batch_norm": cfg['use_batch_norm'],
                    "accuracy": acc_hist,
                    "loss": loss_hist
                }

        except FileNotFoundError:
            print(f"⚠️ Файлы для системы {system} не найдены.")


    for system in systems:
        plt.figure()
        for key, val in results.items():
            if val["system"] == system:
                plt.plot(range(1, num_epochs + 1), val["accuracy"], label=key)
        plt.title(f"Точность {system.title()}")
        plt.xlabel("Эпоха")
        plt.ylabel("Точность")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f"{system}_accuracy.png"))
        plt.close()

        plt.figure()
        for key, val in results.items():
            if val["system"] == system:
                plt.plot(range(1, num_epochs + 1), val["loss"], label=key)
        plt.title(f"Потери {system.title()}")
        plt.xlabel("Эпоха")
        plt.ylabel("Потери")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f"{system}_loss.png"))
        plt.close()

    for cfg in configs:
        plt.figure()
        has_data = False
        for key, val in results.items():
            if (
                val["activation"] == cfg["activation"] and
                val["pooling"] == cfg["pooling"] and
                val["conv_size"] == cfg["conv_size"] and
                val["batch_norm"] == cfg["use_batch_norm"]
            ):
                has_data = True
                plt.plot(range(1, num_epochs + 1), val["accuracy"], label=val["system"])

        if has_data:
            title = f"Accuracy — act={cfg['activation']}, pool={cfg['pooling']}, conv={cfg['conv_size']}, bn={cfg['use_batch_norm']}"
            filename = f"compare_accuracy_{cfg['activation']}_{cfg['pooling']}_conv{cfg['conv_size']}_{'bn' if cfg['use_batch_norm'] else ''}.png"
            plt.title(title)
            plt.xlabel("Эпоха")
            plt.ylabel("Точность")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, filename))
            plt.close()



    for cfg in configs:
        plt.figure()
        has_data = False
        for key, val in results.items():
            if (
                val["activation"] == cfg["activation"] and
                val["pooling"] == cfg["pooling"] and
                val["conv_size"] == cfg["conv_size"] and
                val["batch_norm"] == cfg["use_batch_norm"]
            ):
                has_data = True
                plt.plot(range(1, num_epochs + 1), val["loss"], label=val["system"])

        if has_data:
            title = f"Loss — act={cfg['activation']}, pool={cfg['pooling']}, conv={cfg['conv_size']}, bn={cfg['use_batch_norm']}"
            filename = f"compare_loss_{cfg['activation']}_{cfg['pooling']}_conv{cfg['conv_size']}_{'bn' if cfg['use_batch_norm'] else ''}.png"
            plt.title(title)
            plt.xlabel("Эпоха")
            plt.ylabel("Потери")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, filename))
            plt.close()

    rows = []
    for key, val in results.items():
        for epoch, (acc, loss) in enumerate(zip(val["accuracy"], val["loss"]), start=1):
            rows.append({
                "system": val["system"],
                "activation": val["activation"],
                "pooling": val["pooling"],
                "conv_size": val["conv_size"],
                "batch_norm": val["batch_norm"],
                "epoch": epoch,
                "accuracy": acc,
                "loss": loss
            })

    df = pd.DataFrame(rows)
    df.to_csv("results_summary.csv", index=False)
    df.to_excel("results_summary.xlsx", index=False)
    print("✅ Сводная таблица сохранена в results_summary.csv и .xlsx")


    with open("results_raw.pkl", "wb") as f:
        pickle.dump(results, f)
    print("✅ Полные результаты сохранены в results_raw.pkl")
cd 