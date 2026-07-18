import torch

@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    计算 Linear CKA (Centered Kernel Alignment) 相似度。
    采用特征空间优化算法，比传统 Gram 矩阵计算更快。
    
    Args:
        X: Tensor, 形状 [Batch, Dim] 或 [Batch, Seq, Dim]
        Y: Tensor, 形状 [Batch, Dim] 或 [Batch, Seq, Dim]
        eps: 防止除零的微小量
        
    Returns:
        float: 相似度分数 (0.0 ~ 1.0)
    """
    # 1. 维度处理: 如果是 3D (Batch, Seq, Dim)，展平为 2D (N, Dim)
    if X.dim() > 2: 
        X = X.reshape(-1, X.shape[-1])
    if Y.dim() > 2: 
        Y = Y.reshape(-1, Y.shape[-1])
    
    # 转为 float 保证精度，避免 FP16 溢出
    X = X.float()
    Y = Y.float()

    # 2. 中心化 (Centering): 减去每一列(特征)的均值
    # 对应公式中的 H 矩阵操作
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # 3. 计算分子分母 (利用 Frobenius 范数性质优化)
    # 分子: ||X.T @ Y||_F^2
    # 优化: 计算 DxD 矩阵的范数，而不是 NxN
    numerator = torch.linalg.matrix_norm(X.T @ Y, ord='fro') ** 2
    
    # 分母: ||X.T @ X||_F * ||Y.T @ Y||_F
    norm_xx = torch.linalg.matrix_norm(X.T @ X, ord='fro')
    norm_yy = torch.linalg.matrix_norm(Y.T @ Y, ord='fro')
    
    return float((numerator / (norm_xx * norm_yy + eps)).item())


    # ==========================================
# 简单测试块 (直接运行此文件可测试)
# ==========================================
if __name__ == "__main__":
    print("Testing Linear CKA...")
    
    # 模拟数据: Batch=512, TE_Dim=64, Stable_Dim=128
    N = 512
    x = torch.randn(N, 64)
    y = torch.randn(N, 128)
    
    # 1. 测试随机数据 (应接近 0)
    score_random = linear_cka(x, y)
    print(f"Random Features CKA: {score_random:.4f}")
    
    # 2. 测试完全相同数据 (应等于 1.0)
    score_self = linear_cka(x, x)
    print(f"Identical Features CKA: {score_self:.4f}")
    
    # 3. 测试线性相关数据 (应较高)
    # y = x * W
    y_corr = x @ torch.randn(64, 128)
    score_corr = linear_cka(x, y_corr)
    print(f"Correlated Features CKA: {score_corr:.4f}")