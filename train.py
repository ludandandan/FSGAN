import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    start = time.time()
    opt = TrainOptions().parse() #完成了模型参数，数据集参数等参数的处理，参数的打印保存，gpu启用
    data_loader = CreateDataLoader(opt) #data.CustomDatasetDataLoader(torch.utils.data.DataLoader)，里面有dataloader,dataset(AlignedDataset),opt
    dataset = data_loader.load_data()#载入数据，dataset是data.CustomDatasetDataLoader类型，里面有dataset(包括数据集的绝对路径，数据集内所有照片的绝对路径，风格字典)和dataloader(多线程载入的torch.utils.data.DataLoader，里面载入的就是前面的dataset)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size) # 1057

    model = create_model(opt) # 创建模型，初始化
    model.setup(opt) # 设置学习率调度器，其实学习率不变
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset): # data是一个有23项的dict，包括A,B,A_paths,B_paths,style,style_path,style_name,style_label,style_label_tensor,style_image,style_image_tensor,style_image_path,style_image_name,style_image_label,style_image_label_tensor,style_image_label_tensor_1,style_image_label_tensor_2,style_image_label_tensor_3,style_image_label_tensor_4,style_image_label_tensor_5,style_image_label_tensor_6,style_image_label_tensor_7,style_image_label_tensor_8
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size #本代码设置batch_size=1，所以total_steps每次加1
            epoch_iter += opt.batch_size # epoch_iter同样每次加1，但是每个epoch会清零
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
            
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                #model.save_networks('latest')
                model.save_networks2('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            #model.save_networks('latest')
            #model.save_networks(epoch)
            model.save_networks2('latest')
            model.save_networks2(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    print('Total Time Taken: %d sec' % (time.time() - start))
