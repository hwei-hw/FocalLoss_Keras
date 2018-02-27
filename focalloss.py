def focal_loss(gamma=2, alpha=0.75):
	def focal_loss_fixed(y_true, y_pred):#with tensorflow
		eps = 1e-12
		y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
        	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        	return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
