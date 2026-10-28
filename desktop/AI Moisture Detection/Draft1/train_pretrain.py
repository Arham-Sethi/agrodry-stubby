import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import torch
import matplotlib.pyplot as plt

class MoistureDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        self.data = pd.read_csv(csv_path)
        self.augment = augment
        # general transformation for all datasets
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # set up augmentation for training data
        if self.augment:
            self.transform = transforms.Compose([
                 transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the image path and label from the DataFrame
        row = self.data.iloc[idx]
        img_path = row['image']
        label = float(row['folder'])
        # convert to float tensor, conventionally used for regression tasks
        label = torch.tensor(label, dtype=torch.float32) 
        # convert to conventional RGB Format
        img = Image.open(img_path).convert('RGB') 
        img = self.transform(img)

        return img, label

if __name__ == "__main__":   
    # The dataset is initialized with the path to the CSV file and whether to apply augmentations
    train_dataset = MoistureDataset('train.csv', augment=True)
    val_dataset = MoistureDataset('val.csv', augment=False)
    test_dataset = MoistureDataset('test.csv', augment=False)

    #load the datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    images, labels = next(iter(train_loader))
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")

    # Use GPU if available or else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model chosen is ResNet18, a popular architecture for image classification tasks
    model = models.resnet18(pretrained=True)
    # Means the last layer of the model is replaced with a new linear layer that has 1 output feature
    model.fc = nn.Linear(model.fc.in_features, 1) 
    # This is because we are predicting a single continuous value (moisture level)

    # model is moved to the GPU if available
    model = model.to(device)


    # Mean Squared Error Loss: popular for regression tasks
    criterion = nn.MSELoss() 
    # Adam stands for Adaptive Moment Estimation. It is an advanced optimization algorithm that combines the advantages of: 
    # AdaGrad (adaptive learning rate for each parameter)
    # RMSProp (exponential moving average of gradients)
    # Learning rate: 0.0001 is a common starting point for Adam
    # Betas: (0.9, 0.999) are the coefficients used for computing running averages of gradient and its square
    # eps: 1e-8 is a small constant added to prevent division by zero
    # weight_decay: 0 is used to prevent overfitting by adding a penalty for large weights
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # SCHEDULAR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Mixed precision training setup
    scaler = torch.cuda.amp.GradScaler()

    # Training and Validation Process
    # setting up the no. of times model sees the training dataset
    num_epochs = 20

    # lists to store the training and validation losses
    train_losses = []
    val_losses = []
    early_stop_counter = 0  # For early stopping
    patience = 5  # Number of epochs to wait before stopping

    # Instantiate the best validation loss to a large value
    best_val_loss = float('inf')

    # Loop over the number of epochs
    for epoch in range(1, num_epochs + 1):
        # Set model into training mode
        model.train()
        # Initialize the running loss for the epoch
        runningloss = 0.00
        # Loop over the training data
        for images, labels in train_loader:
            # move both images to the same device  and add a dimension to labels ([32] to [32, 1])
            imgs = images.to(device)
            lbls = labels.to(device).unsqueeze(1)
            # Zero the parameter gradients
            optimizer.zero_grad()
        # Forward pass with mixed precision
            
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = criterion(preds, lbls)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            runningloss += loss.item() * images.size(0)

        # Calculate the average loss for the epoch
        epoch_loss = runningloss / len(train_loader.dataset)
        # Save the training loss to a list for plotting or analysis later.
        train_losses.append(epoch_loss)
            
        #switch to evaluation mode
        model.eval()
        # Initialize the running loss for validation
        runningvalloss = 0.00
        # Loop over the validation data
        with torch.no_grad(): # Temporarily disable gradient calculation to save memory & speed up inference.
            for images, labels in val_loader:
                # move both images to the same device and add a dimension to labels ([32] to [32, 1])
                v_imgs = images.to(device)
                v_lbls = labels.to(device).unsqueeze(1)
                # Forward pass
                v_preds = model(v_imgs)
                # Compute the loss
                val_loss = criterion(v_preds, v_lbls)
                # Add the loss to the running validation loss
                runningvalloss += val_loss.item() * images.size(0)
            # calculate average validation loss for each epoch
            epoch_valloss = runningvalloss / len(val_loader.dataset)
            # Save the validation loss to a list for plotting or analysis later.
            val_losses.append(epoch_valloss)
            # Step the scheduler with the validation loss
            scheduler.step(epoch_valloss)
            # Print the learning rate for the current epoch
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
            # Print the training and validation loss for the epoch
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val loss: {epoch_valloss:.4f}")
        
        # Early stopping and model saving
        if epoch_valloss < best_val_loss:
            best_val_loss = epoch_valloss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

    # TEST TIME!!
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    # Initialize the running loss for the test set
    testrunningloss = 0.00
    # Loop over the test data
    with torch.no_grad():
        for images, labels in test_loader:
            t_imgs = images.to(device)
            t_lbls = labels.to(device).unsqueeze(1)
            t_preds = model(t_imgs)
            test_loss = criterion(t_preds, t_lbls)
            testrunningloss += test_loss.item() * images.size(0)
        testrunningloss /= len(test_loader.dataset)
        print(f"Test Loss: {testrunningloss:.4f}")


    # Plotting the training and validation losses
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'final_model.pth')
    # Save the training and validation losses to a CSV file
    losses_df = pd.DataFrame([
        {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1)
    ])
        

    losses_df.to_csv('losses.csv', index=False)

    # Generate predictions on the test set
    preds_list = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images).cpu().numpy().flatten().tolist()
            preds_list.extend(outputs)

    # Save to CSV
    pd.DataFrame({'predicted_moisture': preds_list}).to_csv('test_predictions.csv', index=False)


    # Save the model architecture and state dictionary
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_epochs': num_epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, 'model_checkpoint.pth')



