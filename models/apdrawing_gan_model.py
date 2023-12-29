import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

class APDrawingGANModel(BaseModel):
    def name(self):
        return 'APDrawingGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')# no_lsgan=True, use_lsgan=False 设置默认值
        parser.set_defaults(dataset_mode='aligned')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt) # 完成基础模型的初始化设置，包括gpu，存储路径等
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_VGG', 'G_CLS', 'G_local'] #具体要打印的训练Loss
        self.loss_names.append('D_real_local')
        self.loss_names.append('D_fake_local')
        self.loss_names.append('G_GAN_local')
        if self.isTrain and self.opt.no_l1_loss:
            self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'D']

        self.loss_names.append('G')
        print('loss_names', self.loss_names)
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B'] # 想要保存和展示的图像名字
        if self.opt.use_local:
            self.visual_names += ['fake_B1']
        if not self.isTrain and self.opt.save2:
            self.visual_names = ['real_A', 'fake_B']
        print('visuals', self.visual_names)
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain: #想要保存的模型
            self.model_names = ['D', 'D_Cls']
        else:  # during test time, only load Gs 
            self.model_names = ['G']
            self.auxiliary_model_names = []
        if self.opt.use_local: #如果使用局部的GAN，还需要保存这些两个眼睛，一个鼻子，一个嘴巴，一个头发，一个剩下的，一个总的结合的
            self.model_names += ['GLEyel','GLEyer','GLNose','GLMouth','GLHair','GLBG','GCombine']
        print('model_names', self.model_names)

        if self.isTrain:
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            use_sigmoid = opt.no_lsgan #True
            # opt.input_nc=3, opt.ndf*2=64,opt.n_layers_D=3,opt.norm=batch,use_sigmoid=True,opt.init_type='kaiming',opt.init_gain=0.02, self.gpu_ids=1,2
            self.netD_Cls = networks.define_D(opt.input_nc, opt.ndf*2, 'basic_cls', 1, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',2, # 3+3,32,'multiscale',2,3,'batch',True,'kaiming',0.02,[1,2]
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            print('netD', opt.netD, opt.n_layers_D)
            
            if self.opt.discriminator_local:
                self.netDLEyel = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLEyer = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLNose = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLMouth = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLHair = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLBG = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.opt.use_local:
            self.netGLEyel = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm, #3,3,64,'global','batch',True,'kaiming',0.02,[1,2]
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLEyer = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLNose = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLMouth = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLHair = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLBG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGCombine = networks.define_G(opt.output_nc, opt.output_nc, opt.ngf, 'local', opt.norm,#3,3,64,'local','batch',True,'kaiming',0.02,[1,2]
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size) # opt.pool_size=0
            # define loss functions
            self.criterionCls = torch.nn.CrossEntropyLoss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.TVloss = networks.TVLoss().to(self.device)
            self.criterionVGG = networks.VGGLoss(self.device)

            # initialize optimizers
            self.optimizers = []
            G_params = list(self.netGLEyel.parameters()) + list(self.netGLEyer.parameters()) + list(self.netGLNose.parameters()) + list(self.netGLMouth.parameters()) + list(self.netGLHair.parameters()) + list(self.netGLBG.parameters()) + list(self.netGCombine.parameters()) 
            print('G_params 8 components')
            self.optimizer_G = torch.optim.Adam(G_params,
                                                lr=opt.lr, betas=(opt.beta1, 0.999))##opt.lr=0.0002, opt.beta1=0.5
            if not self.opt.discriminator_local:
                print('D_params 1 components')
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr * 0.01, betas=(opt.beta1, 0.999)) #opt.lr=0.0002, opt.beta1=0.5
            else:
                D_params = list(self.netD.parameters()) + list(self.netDLEyel.parameters()) +list(self.netDLEyer.parameters()) + list(self.netDLNose.parameters()) + list(self.netDLMouth.parameters()) + list(self.netDLHair.parameters()) + list(self.netDLBG.parameters())
                print('D_params 7 components')
                self.optimizer_D = torch.optim.Adam(D_params,
                                                lr=opt.lr * 0.01, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # 真实照片[1,3,512,512]，数值在-1到1之间
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # 真实素描[1,3,512,512]，数值在-1到1之间
        self.real_B_style = input['style'].to(self.device) # 真实素描的风格[1,3,256,256]，每个像素是[0,0,1]表示风格类别是2
        self.real_B_label = input['label'].to(self.device) # 真实素描风格的标签，就一个数字，表示风格类别
        self.image_paths = input['A_paths' if AtoB else 'B_paths'] # 真实照片的路径
        if self.opt.use_local:
            self.real_A_eyel = input['eyel_A'].to(self.device) # 真实照片的左眼区域[1,3,80,112]，数值在-1到1之间
            self.real_A_eyer = input['eyer_A'].to(self.device) # 真实照片的右眼区域[1,3,80,112]，数值在-1到1之间
            self.real_A_nose = input['nose_A'].to(self.device) # 真实照片的鼻子区域[1,3,96,96]，数值在-1到1之间
            self.real_A_mouth = input['mouth_A'].to(self.device) # 真实照片的嘴巴区域[1,3,80,128]，数值在-1到1之间
            self.real_B_eyel = input['eyel_B'].to(self.device) # 真实素描的左眼区域[1,3,80,112]，数值在-1到1之间
            self.real_B_eyer = input['eyer_B'].to(self.device) # 真实素描的右眼区域[1,3,80,112]，数值在-1到1之间
            self.real_B_nose = input['nose_B'].to(self.device) # 真实素描的鼻子区域[1,3,96,96]，数值在-1到1之间
            self.real_B_mouth = input['mouth_B'].to(self.device) # 真实素描的嘴巴区域[1,3,80,128]，数值在-1到1之间
            self.center = input['center'] #[4,2],左眼，右眼，鼻子，嘴巴的中心点坐标
            self.real_A_hair = input['hair_A'].to(self.device) # [1,3,512,512]，除背景，眼睛鼻子嘴巴区域的其他区域的照片，数值在-1到1之间，背景眼睛鼻子嘴巴区域是-1
            self.real_B_hair = input['hair_B'].to(self.device) # [1,3,512,512]，除背景，眼睛鼻子嘴巴区域的其他区域的素描，数值在-1到1之间，背景眼睛鼻子嘴巴区域是-1
            self.real_A_bg = input['bg_A'].to(self.device) # [1,3,512,512]，背景区域的照片，数值在-1到1之间，其他区域是-1
            self.real_B_bg = input['bg_B'].to(self.device) # [1,3,512,512]，背景区域的素描，数值在-1到1之间，其他区域是-1
            self.mask = input['mask'].to(self.device) # mask for non-eyes,nose,mouth #[1,3,512,512],眼睛鼻子嘴巴区域是0，其他区域是1
            self.mask2 = input['mask2'].to(self.device) # mask for non-bg#[1,3,512,512],背景区域是0，其他区域是1
        

    def forward(self):
        # EYES, NOSE, MOUTH
        fake_B_eyel = self.netGLEyel(self.real_A_eyel) # [1,3,80,112]
        fake_B_eyer = self.netGLEyer(self.real_A_eyer) # [1,3,80,112]
        fake_B_nose = self.netGLNose(self.real_A_nose) # [1,3,96,96]
        fake_B_mouth = self.netGLMouth(self.real_A_mouth) # [1,3,80,128]
        self.fake_B_nose = fake_B_nose
        self.fake_B_eyel = fake_B_eyel
        self.fake_B_eyer = fake_B_eyer
        self.fake_B_mouth = fake_B_mouth
            
        # HAIR, BG AND PARTCOMBINE
        fake_B_hair = self.netGLHair(self.real_A_hair) # [1,3,512,512]
        fake_B_bg = self.netGLBG(self.real_A_bg) # [1,3,512,512]
        self.fake_B_hair = self.masked(fake_B_hair,self.mask*self.mask2)#由于self.mask是除眼睛鼻子嘴巴区域外的区域为1，self.mask2是人头区域是1，所以他俩相乘就是人头上除眼睛鼻子嘴巴之外的区域被保留了（也就是头发和部分面部），其他区域为0
        self.fake_B_bg = self.masked(fake_B_bg,self.inverse_mask(self.mask2))#由于self.mask2是人头区域是1，经过翻转之后就是背景区域是1，这样就背景区域被保留了，其他区域为0
        self.fake_B1 = self.partCombiner2_bg(fake_B_eyel,fake_B_eyer,fake_B_nose,fake_B_mouth,fake_B_hair,fake_B_bg,self.mask*self.mask2,self.inverse_mask(self.mask2),self.opt.comb_op)
            
        # FUSION NET
        self.fake_B = self.netGCombine(self.fake_B1, self.real_B_style)

        
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # 将真的照片图与假的素描cat之后，进行self.fake_AB_pool.query方法，由于此前定义的pool_size=0，所以直接返回cat之后的图，[1,6,512,512]
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1)) # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD.forward(fake_AB.detach())#将上面得到的[1,6,512,512]输入到判别器中, 返回的结果是10个特征值图,[list1,list2],每个list里面有5个特征图
        _, pred_real_cls = self.netD_Cls(self.real_B)#pred_real_cls是[1,3]表示预测的风格类别
        loss_D_real_cls = self.criterionCls(pred_real_cls, self.real_B_label) #预测的风格类别与真实的风格类别的交叉熵损失
        self.loss_D_fake = self.criterionGAN(pred_fake, False)# 计算对抗损失（使用patch GAN的思想，计算若干特征图与0tensor之间的MSEloss，如果没有判别为假就进行惩罚）

        if self.opt.discriminator_local:
            fake_AB_parts = self.getLocalParts(fake_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_D_fake_local = 0
            for i in range(len(fake_AB_parts)):
                net = getattr(self, 'net' + local_names[i]) # 获取每个局部判别器
                pred_fake_tmp = net.forward(fake_AB_parts[i].detach())# 前向计算，获得每个局部判别器的输出
                addw = self.getaddw(local_names[i]) # 获取每个局部判别器的权重，只有头发是1.8，其他都是1.0
                self.loss_D_fake_local = self.loss_D_fake_local + self.criterionGAN(pred_fake_tmp, False) * addw # 计算每个局部判别器的对抗损失并加权求和
            self.loss_D_fake = self.loss_D_fake + self.loss_D_fake_local# 将全局判别器和局部判别器的对抗损失求和

        # Real
        _, pred_fake_cls = self.netD_Cls(self.fake_B.detach()) #假的素描输入到风格分类器中，得到预测的风格类别
        loss_D_fake_cls = self.criterionCls(pred_fake_cls, self.real_B_label) 
        real_AB = torch.cat((self.real_A, self.real_B), 1) #真照片和真素描cat在一起，[1,6,512,512]
        pred_real = self.netD.forward(real_AB) #真照片和真素描输入判别器中
        self.loss_D_real = self.criterionGAN(pred_real, True)#计算2个模块最后1个特征图与1tensor之间的MSEloss【如果没有判别为真就进行惩罚】
        if self.opt.discriminator_local: # 计算局部判别器的对抗损失
            real_AB_parts = self.getLocalParts(real_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_D_real_local = 0
            for i in range(len(real_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_real_tmp = net.forward(real_AB_parts[i])
                addw = self.getaddw(local_names[i])
                self.loss_D_real_local = self.loss_D_real_local + self.criterionGAN(pred_real_tmp, True) * addw
            self.loss_D_real = self.loss_D_real + self.loss_D_real_local # 将全局判别器和局部判别器的对抗损失求和
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + (loss_D_real_cls + loss_D_fake_cls) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) # 将真实照片和假素描cat在一起，[1,6,512,512]
        pred_fake = self.netD.forward(fake_AB) #经过判别器计算得到10个特征图，[list1,list2],每个list里面有5个特征图
        real_AB = torch.cat((self.real_A, self.real_B), 1) # 将真实照片和真实素描cat在一起，[1,6,512,512]
        pred_real = self.netD.forward(real_AB) #经过判别器计算得到10个特征图，[list1,list2],每个list里面有5个特征图
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)#计算对抗损失，计算真照片-假素描计算的10个特征图与1tensor之间的MSEloss【如果没有判别为真就进行惩罚】
        self.loss_G_GAN_local = 0 # 局部判别器的对抗损失，权重是1
        self.loss_G_local = 0 #这里首先存的是特征匹配损失，第k个局部判别器的【真照片-真素描】特征图与【真照片-假素描】特征图的L1loss，权重是25
        ###local feat loss
        if self.opt.discriminator_local: # 使用局部判别器
            fake_AB_parts = self.getLocalParts(fake_AB) #获取真实照片假素描的局部区域
            real_AB_parts = self.getLocalParts(real_AB) #获取真实照片真素描的局部区域
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            for i in range(len(fake_AB_parts)):
                net = getattr(self, 'net' + local_names[i])# 获取局部判别网络
                pred_fake_tmp = net.forward(fake_AB_parts[i])# 前向计算，真照片-假素描输入局部判别器，输出[list1,list2],每个list里面有5个特征图
                pred_real_tmp = net.forward(real_AB_parts[i])# 前向计算，真照片-真素描输入局部判别器，输出[list1,list2],每个list里面有5个特征图
                addw = self.getaddw(local_names[i]) # 获取局部判别器的权重，只有头发是1.8，其他都是1.0
                self.loss_G_GAN_local = self.loss_G_GAN_local + self.criterionGAN(pred_fake_tmp, True) * addw # 计算对抗损失，计算真照片-假素描计算的10个特征图与1tensor之间的MSEloss【如果没有判别为真就进行惩罚】
                if self.opt.use_local and not self.opt.no_G_local_loss:
                    feat_weights=4.0/4.0 #1.0
                    D_weights = 1.0 / 1.0 #1.0
                    for k in range(1): #这里计算的是局部特征匹配损失
                        for j in range(len(pred_fake_tmp[k])-1):#range(4)，将计算32，64，128，256这4个通道特征图（真实与假的的L1loss），乘以权重25，对头发再乘以1.8
                            self.loss_G_local += D_weights * feat_weights * self.criterionL1(pred_fake_tmp[k][j], pred_real_tmp[k][j].detach()) * self.opt.lambda_local * addw 
        
        ###local l1 vgg loss### 这里再self.loss_G_local加上局部感知损失（就是真素描和假素描通过VGG的特征图们计算的L1loss），权重是12.5=25*0.5
        if self.opt.use_local and not self.opt.no_G_local_loss:
            local_names = ['eyel','eyer','nose','mouth','hair','bg']
            for i in range(len(local_names)):
                fakeblocal = getattr(self, 'fake_B_' + local_names[i]) #局部假素描
                realblocal = getattr(self, 'real_B_' + local_names[i]) #局部真素描
                addw = self.getaddw(local_names[i]) #权重
                self.loss_G_local += self.criterionVGG(fakeblocal,realblocal)* self.opt.lambda_local * addw * 0.5 #局部感知损失
                self.loss_G_local += self.criterionL1(fakeblocal,realblocal)* self.opt.lambda_local * addw #局部逐像素损失
        
        ###global l1 vgg feat loss#### 全局的感知损失（VGG）和全局的逐像素损失（L1）
        if not self.opt.no_l1_loss:#self.loss_G_VGG 计算原尺寸，缩小到0.25倍的尺寸和缩小到0.5倍的尺寸的VGG损失,将这3个损失相加，权重是12.5/3
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 # 这里是逐像素损失，真照片-假素描的L1loss，权重是25，这一步是计算图像像素空间的
            self.loss_G_VGG = (self.criterionVGG(self.fake_B, self.real_B) + self.criterionVGG(F.interpolate(self.fake_B, scale_factor=0.25, recompute_scale_factor=True), F.interpolate(self.real_B, scale_factor=0.25, recompute_scale_factor=True)) + self.criterionVGG(F.interpolate(self.fake_B, scale_factor=0.5, recompute_scale_factor=True), F.interpolate(self.real_B, scale_factor=0.5, recompute_scale_factor=True)))  *self.opt.lambda_L1 * 0.5/ 3.0
            feat_weights=4.0/4.0
            D_weights = 1.0 / 2.0 #0.5
            #self.loss_G_L1 = 0
            for i in range(2): # 这里计算的是全局的特征匹配损失
                for j in range(len(pred_fake[i])-1): #pred_fake是真照片-假素描经过判别器计算得到10个特征图，[list1,list2],每个list里面有5个特征图，这里是计算特征空间的逐像素损失
                    self.loss_G_L1 += 1.0 * D_weights * feat_weights * self.criterionL1(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_L1
            
        _, pred_fake_cls = self.netD_Cls(self.fake_B) #假的素描输入到风格分类器中，得到预测的风格类别
        self.loss_G_CLS = self.criterionCls(pred_fake_cls, self.real_B_label)    #计算风格分类器的交叉熵损失  

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_local #全局对抗损失和局部对抗损失，权重都是1
        if 'G_L1' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_L1 # 全局的特征匹配损失+逐像素损失
        if 'G_VGG' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_VGG # 全局的感知损失，在3个尺寸上输入VGG得到特征图计算
        if 'G_CLS' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_CLS # 风格分类器的交叉熵损失（假素描的风格与真实风格的交叉熵）
        if 'G_local' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_local #局部特征匹配损失+局部感知损失+局部逐像素损失


        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D 更新判别器
        self.set_requires_grad(self.netD, True) # enable backprop for D 将netD中参数的requires_grad设置为True
        if self.opt.discriminator_local:
            self.set_requires_grad(self.netDLEyel, True)
            self.set_requires_grad(self.netDLEyer, True)
            self.set_requires_grad(self.netDLNose, True)
            self.set_requires_grad(self.netDLMouth, True)
            self.set_requires_grad(self.netDLHair, True)
            self.set_requires_grad(self.netDLBG, True)
        self.optimizer_D.zero_grad() # 清空判别器参数之前的梯度
        self.backward_D() # 计算梯度
        self.optimizer_D.step() # 判别器参数更新

        # update G
        self.set_requires_grad(self.netD, False) # D requires no gradients when optimizing G 将netD中参数的requires_grad设置为False，在优化G的时候不需要计算判别器的梯度
        if self.opt.discriminator_local:
            self.set_requires_grad(self.netDLEyel, False)
            self.set_requires_grad(self.netDLEyer, False)
            self.set_requires_grad(self.netDLNose, False)
            self.set_requires_grad(self.netDLMouth, False)
            self.set_requires_grad(self.netDLHair, False)
            self.set_requires_grad(self.netDLBG, False)
        self.optimizer_G.zero_grad() # 清空生成器参数之前的梯度
        self.backward_G() # 计算梯度
        self.optimizer_G.step() # 生成器参数更新
