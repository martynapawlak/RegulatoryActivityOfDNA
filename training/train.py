
from torchmetrics import Accuracy, R2Score


train_path = "/content/drive/MyDrive/data/project1DL/train.tsv"
val_path = "/content/drive/MyDrive/data/project1DL/val.tsv"

def training_loop_CNN(train_path, val_path, epochs=100, lr=0.0001, batch_size=64):

  train_df = pd.read_csv(train_path, sep="\t")

  mean_reg = train_df["rna_dna_ratio"].mean()
  std_reg = train_df["rna_dna_ratio"].std()

  train_dataset = DNADataset(train_path, mean_reg, std_reg, augment=True)
  val_dataset = DNADataset(val_path, mean_reg, std_reg, augment=False)

  train_losses = []
  valid_losses = []

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size)

  # model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = DNAMultitaskModel().to(device)

  # loss function: BCE - classification (0/1), MSE - regression (number) - we add those two
  criterion_class = nn.BCELoss()
  criterion_reg = nn.MSELoss()

  optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

  # metrics initialization
  accuracy_metric = Accuracy(task="binary").to(device) 
  r2_metric = R2Score().to(device)

  # training loop

  best_val_loss = float("inf")

  for epoch in range(epochs):
    model.train()
    train_loss = 0

    for seq, y_class, y_reg in train_loader:
      seq = seq.to(device)
      y_class = y_class.to(device)
      y_reg = y_reg.to(device)

      optimizer.zero_grad()

      out_class, out_reg = model(seq)

      loss_class = criterion_class(out_class.squeeze(), y_class)
      loss_reg = criterion_reg(out_reg.squeeze(), y_reg)

      loss = loss_class + 0.1 * loss_reg 
      loss.backward()
      optimizer.step()

      train_loss += loss.item()

    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0

    with torch.no_grad():
      for seq, y_class, y_reg in val_loader:
        seq = seq.to(device)
        y_class = y_class.to(device)
        y_reg = y_reg.to(device)

        out_class, out_reg = model(seq)

        loss_class = criterion_class(out_class.squeeze(), y_class)
        loss_reg = criterion_reg(out_reg.squeeze(), y_reg)

        loss = loss_class + 0.1 * loss_reg

        val_loss += loss.item()

        preds_class = (out_class.squeeze() > 0.5).float()               
        accuracy_metric.update(preds_class, y_class)
        r2_metric.update(out_reg.squeeze(), y_reg)

    val_loss = val_loss / len(val_loader)
    valid_losses.append(val_loss)

    if val_loss < best_val_loss:
    	best_val_loss=val_loss 
		torch.save(model.state_dict(), 'dna_model.pt')
		print(f"best new model saved !! :)")

    # computing the metrics
    val_acc = accuracy_metric.compute()
    val_r2 = r2_metric.compute()

    print(f"epoch: {epoch+1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc*100:.1f}, val_r2: {val_r2:.4f}")
    print(f"BCE (class): {loss_class.item():.4f}, MSE (reg): {loss_reg.item():.4f}")

    accuracy_metric.reset()
    r2_metric.reset()

  return model, train_losses, valid_losses


model,train_losses, valid_losses = training_loop_CNN(train_path, val_path)


# simple plot
plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()


