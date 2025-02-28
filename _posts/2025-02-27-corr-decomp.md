---
title: "Correlation Attribution is Exposure Times Risk Times Correlation Divided by Portfolio Risk"
date: 2025-02-27T01:20:00-04:00
categories:
  - Blog
tags:
  - X-sigma-rho
  - Correlation
  - Portfolio Construction
  - Deepseek R1
---

---

## TL;DR

Two portfolios share common holdings. You are curious which holdings are largest contributors to correlation of the two returns. The result below, assisted by Deepseek R1, tells you exactly how to decompose correlation in an additive and non-overlapping way. You may be surprised by the fact that even non-common holdings can contribute significantly to correlation through indirect channels.

**Final Simplified Formula:**  
\[
\text{Contribution}_{\rho}(k) = \frac{\sigma_k}{2} \left( \frac{w_{A_k}}{\sigma_A} \rho_{r_B, r_k} + \frac{w_{B_k}}{\sigma_B} \rho_{r_A, r_k} \right)
\]  

**Symbol Definitions**:  

- \(\sigma_k\): Volatility of **asset \(k\)'s returns** (\(r_k\)).  
- \(w_{A_k}\), \(w_{B_k}\): Weight of asset \(k\) in portfolios \(A\) and \(B\), respectively.  
- \(\sigma_A\), \(\sigma_B\): Volatilities of **portfolio \(A\)'s returns** (\(r_A\)) and **portfolio \(B\)'s returns** (\(r_B\)).  
- \(\rho_{r_B, r_k}\): Correlation between portfolio \(B\)'s returns (\(r_B\)) and asset \(k\)'s returns (\(r_k\)).  
- \(\rho_{r_A, r_k}\): Correlation between portfolio \(A\)'s returns (\(r_A\)) and asset \(k\)'s returns (\(r_k\)).  

---

**Derivation Walkthrough:**  

1. **Euler Theorem for Bilinear Covariance**:  
   Covariance \(\text{Cov}(r_A, r_B)\) is a **bilinear function** (homogeneous of degree 2 in portfolio weights). Scaling both portfolios by \(t\):  
   \[
   \text{Cov}(t r_A, t r_B) = t^2 \cdot \text{Cov}(r_A, r_B)
   \]  
   By Euler’s theorem for degree \(k=2\):  
   \[
   \sum_{k=1}^N \left( w_{A_k} \cdot \frac{\partial \text{Cov}(r_A, r_B)}{\partial w_{A_k}} + w_{B_k} \cdot \frac{\partial \text{Cov}(r_A, r_B)}{\partial w_{B_k}} \right) = 2 \cdot \text{Cov}(r_A, r_B)
   \]  
   To resolve double-counting, we introduce a \(1/2\) adjustment factor:  
   \[
   \text{Contribution}(k) = \frac{1}{2} \left( w_{A_k} \cdot \frac{\partial \text{Cov}(r_A, r_B)}{\partial w_{A_k}} + w_{B_k} \cdot \frac{\partial \text{Cov}(r_A, r_B)}{\partial w_{B_k}} \right)
   \]  
   _Intuition_: The factor \(1/2\) fairly splits covariance contributions between portfolios \(A\) and \(B\).  

2. **Express Partial Derivatives**:  
   Substitute \(\frac{\partial \text{Cov}(r_A, r_B)}{\partial w_{A_k}} = \text{Cov}(r_B, r_k)\) and \(\frac{\partial \text{Cov}(r_A, r_B)}{\partial w_{B_k}} = \text{Cov}(r_A, r_k)\):  
   \[
   \text{Contribution}_{\text{Cov}}(k) = \frac{w_{A_k} \cdot \text{Cov}(r_B, r_k) + w_{B_k} \cdot \text{Cov}(r_A, r_k)}{2}
   \]  

3. **Convert to Correlation**:  
   Divide by \(\sigma_A \sigma_B\) to scale contributions to correlation:  
   \[
   \text{Contribution}_{\rho}(k) = \frac{\text{Contribution}_{\text{Cov}}(k)}{\sigma_A \sigma_B}
   \]  

4. **Substitute Covariance with Correlation**:  
   Using \(\text{Cov}(r_B, r_k) = \rho_{r_B, r_k} \sigma_B \sigma_k\) and \(\text{Cov}(r_A, r_k) = \rho_{r_A, r_k} \sigma_A \sigma_k\):  
   \[
   \text{Contribution}_{\rho}(k) = \frac{\sigma_k}{2} \left( \frac{w_{A_k} \rho_{r_B, r_k}}{\sigma_A} + \frac{w_{B_k} \rho_{r_A, r_k}}{\sigma_B} \right)
   \]  

---

**Key Takeaways**:  

- **Dual Dependence**: Contributions depend on correlations between **both portfolios** and asset \(k\).  
- **Volatility Scaling**: Higher asset volatility (\(\sigma_k\)) amplifies impact.  
- **Normalized Weights**: Weights are scaled by portfolio volatilities (\(\sigma_A, \sigma_B\)), penalizing riskier allocations.  

This formula quantifies how shared holdings drive portfolio correlation, enabling precise adjustments for diversification or systemic risk management.  

Best regards,  
[Your Name]  

---

Copy-paste this into any markdown-supported editor for proper rendering.

