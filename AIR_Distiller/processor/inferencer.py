import os
import torch
from utils.metrics import R1_mAP_eval
import logging


def inference(
        cfg,
        distiller,
        query_loader,
        gallery_loader
):
    device = "cuda"
    logger = logging.getLogger("Asymmetric_Image_Retrieval.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(max_rank=100, metric=cfg.TEST.TEST_METRIC, reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    distiller.eval()
    for n_iter, (img, pid, camid) in enumerate(query_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = distiller.forward_query(image=img)
                    feat = feat + f
            else:
                feat = distiller.forward_query(image=img)

            evaluator.query_update((feat, pid, camid))

    for n_iter, (img, pid, camid) in enumerate(gallery_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = distiller.forward_gallery(image=img)
                    feat = feat + f
            else:
                feat = distiller.forward_gallery(image = img)

            evaluator.gallery_update((feat, pid, camid))

    cmc, mAP, mINP = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))

    output_dir = os.path.join(cfg.OUTPUT_DIR.ROOT_PATH, cfg.OUTPUT_DIR.EXPERIMENT_NAME)

    with open('%s/test_acc.txt' % output_dir, 'a') as test_file:
        if cfg.TEST.RE_RANKING:
            test_file.write('[With Re-Ranking] mAP: {:.2%}, mINP: {:.2%}, rank1: {:.2%} rank5: {:.2%} rank10: {:.2%}\n'
                            .format(mAP, mINP, cmc[0], cmc[4], cmc[9]))


        #########################no re rank##########################
        else:
            test_file.write(
                '[Without Re-Ranking]mAP: {:.2%}, mINP: {:.2%}, rank1: {:.2%} rank5: {:.2%} rank10: {:.2%}\n'
                .format(mAP, mINP, cmc[0], cmc[4], cmc[9]))
            

   