# 実装しながらMD_simulationを理解する

## 1. MDシミュレーションの基本の実装
ハミルトニアン系でのMDシミュレーションの最も簡単な実装  
実装（simple_MD.py）には以下を含む  
- 2次元のシミュレーションボックス
- 粒子数は10程度
- 周期境界条件の導入
- 座標・運動量での記載
- Lennard-Jonesポテンシャルでの相互作用記述
- Velocity Verlet法での更新

  
これに紐付けして温度や体積制御、拘束条件付きのダイナミクスなどの勉強を進める  

![md_animation](https://github.com/user-attachments/assets/8e12e37b-71f0-48e0-885d-ef6caabcdfbf)
