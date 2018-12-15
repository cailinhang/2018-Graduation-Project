from torchvision import datasets, transforms
import torch

def load_dataset(dataset='cifar10',train_batch_size=2,test_batch_size=16, kwargs={}, train=True ):
    if dataset == 'cifar10':
        if train == True:
           # little datasets  截取小数据集
            dataset=datasets.CIFAR10('../L1_norm/data.cifar10', train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.Pad(4),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                               ])) 
        
            dataset.train_data = dataset.train_data[:20]
            
            train_loader = torch.utils.data.DataLoader(
                # datasets[0][0] 是 三通道图片 torch.Size([3, 40, 40])
                # datasets[0][1] 是 标签 0~9       
                dataset,
                batch_size=train_batch_size, shuffle=True, **kwargs)
        
        # 测试集
        dataset = datasets.CIFAR10('../L1_norm/data.cifar10', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ]))
        dataset.test_data = dataset.test_data[:50]
        
        test_loader = torch.utils.data.DataLoader(
                dataset,
            batch_size=test_batch_size, shuffle=True, **kwargs) 
    else:
        # cifar100 train data
        if train == True:
            dataset = datasets.CIFAR100('./data.cifar100', train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.Pad(4),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                               ])
            )
            
            dataset.train_data = dataset.train_data[:20]
            
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=train_batch_size, shuffle=True, **kwargs)
        
        
        # cifar-100 test data
        
        dataset = datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ]))
        dataset.test_data = dataset.test_data[:50]
        
        test_loader = torch.utils.data.DataLoader(
                dataset,
            batch_size=test_batch_size, shuffle=True, **kwargs)
    
    if train == False:
        return test_loader
    
    return train_loader, test_loader        

if __name__=='__main__':
    train_loader, test_loader = load_dataset()
        