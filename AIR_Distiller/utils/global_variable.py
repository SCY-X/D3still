from config import cfg



class global_varible():
    def __init__(self, block_type, deploy_flag):
        assert block_type in ['base', 'DBB', 'WB']
        self.CONV_BN_TYPE = block_type
        self.DEPLOY_FLAG = deploy_flag

    def block_type(self):
        return self.CONV_BN_TYPE
    def deploy_flag(self):
        return self.DEPLOY_FLAG
