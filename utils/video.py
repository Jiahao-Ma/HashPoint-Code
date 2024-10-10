from utils.utils import imgs2gif
import os

# train_path = r'exp/rmsprop/train'
# train_paths = [os.path.join(train_path, p) for p in os.listdir(train_path)]
# imgs2gif(train_paths, saveName=os.path.join(train_path, 'train.gif'))

test_path = r'exp/rmsprop/test'
test_paths = [os.path.join(test_path, f"{i:02d}.png") for i in range(200)]
imgs2gif(test_paths, saveName=os.path.join(test_path, 'test.gif'), duration=0.8)
