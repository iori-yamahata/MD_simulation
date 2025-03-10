import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation

class SimpleMD:
    def __init__(self, n_particles=2, dim=2, dt=0.01, box_size=10.0):
        self.n_particles = n_particles          # 粒子数
        self.dim         = dim                  # 今回は2次元シミュレーション
        self.dt          = dt                   # シミュレーションの時間ステップ幅
        self.box_size    = box_size             # シミュレーションボックスのサイズ
        self.mass        = np.ones(n_particles) # 簡単のために粒子の質量は全て1にする

        # ハミルトニアン形式での実装
        # 粒子の座標ベクトルと運動量ベクトル（位相空間）を初期化
        # 2次元出のシミュレーションのため、1粒子に対して4つ（x座標、y座標、x運動量、y運動量）の情報を持つ
        self.positions   = generate_positions(n_particles, dim, box_size, min_dist=1.0) # 最下部の補助関数を利用しています。
        self.momenta     = np.random.uniform(-1, 1, (n_particles, dim)) # ランダム生成だが、ボルツマン分布からサンプリングするのが普通
    
    def kinetic_energy(self):
        # 運動エネルギーを計算
        # これにより、システムの運動エネルギーの総和を得る
        return np.sum(np.sum(self.momenta**2, axis=1) / (2 * self.mass))
    
    def potential_energy(self):
        # ポテンシャルエネルギーを計算
        # ポテンシャルはLennard-Jonesポテンシャルを用いてポテンシャルエネルギーの総和を得る
        epsilon, sigma = 1.0, 1.0
        potential = 0.0
        
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles): # 距離は対称なので、重複を避ける
                r_vec = self.positions[i] - self.positions[j]
                r_vec = r_vec - self.box_size * np.round(r_vec / self.box_size) # 周期境界条件を考慮
                r     = np.sqrt(np.sum(r_vec ** 2)) # 粒子間距離
                
                if r > 0:
                    sr6        = (sigma / r) ** 6
                    sr12       = sr6 ** 2
                    potential += 4 * epsilon * (sr12 - sr6)  # Lennard-Jonesポテンシャルを足す
        
        return potential
    
    def forces(self):
        forces = np.zeros((self.n_particles, self.dim))
        epsilon, sigma = 1.0, 1.0
        
        for i in range(self.n_particles):
            for j in range(self.n_particles): # 全ての粒子組み合わせに対して計算
                if i != j:
                    r_vec = self.positions[i] - self.positions[j]
                    r_vec = r_vec - self.box_size * np.round(r_vec / self.box_size)
                    r     = np.sqrt(np.sum(r_vec ** 2))
                    
                    if r > 0:
                        sr6        = (sigma / r) ** 6
                        sr12       = sr6 ** 2
                        f_scalar   = 24 * epsilon * (2 * sr12 - sr6) / r ** 2 # Lennard-Jonesポテンシャルの微分
                        forces[i] += f_scalar * r_vec # 力を計算

        return forces
    
    def velocity_verlet_step(self):
        # Velocity Verlet法による1ステップの計算（GROMACSではleap frog法？）
        # 位置と運動量を更新する
        # Verletはシンプレクティック積分法であるため、エネルギーが保存される（勉強中）
        forces_current  = self.forces()
        self.positions += self.momenta / self.mass[:, np.newaxis] * self.dt + forces_current / (2 * self.mass[:, np.newaxis]) * self.dt**2 # 位置の更新
        self.positions  = self.positions % self.box_size # 周期境界条件を考慮
        forces_new      = self.forces()
        self.momenta   += (forces_current + forces_new) / 2 * self.dt # 運動量の更新
        
        return {
            'kinetic'   : self.kinetic_energy(),
            'potential' : self.potential_energy(),
            'total'     : self.kinetic_energy() + self.potential_energy()
        }


def run_simulation(n_particles=4, n_steps=1000, dt=0.01, save_animation=True):
    # シミュレーション設定
    box_size = 10.0
    md       = SimpleMD(n_particles = n_particles, 
                        dim         = 2, 
                        dt          = dt, 
                        box_size    = box_size)
    
    # データ記録用のものたち
    positions_history = np.zeros((n_steps + 1, n_particles, 2))
    energy_history = {
        'kinetic'   : np.zeros(n_steps + 1),
        'potential' : np.zeros(n_steps + 1),
        'total'     : np.zeros(n_steps + 1),
        'time'      : np.zeros(n_steps + 1)
    }
    
    # 初期状態を記録
    positions_history[0] = md.positions.copy()
    energy = {
        'kinetic'  : md.kinetic_energy(),
        'potential': md.potential_energy(),
        'total'    : md.kinetic_energy() + md.potential_energy()
    }
    energy_history['kinetic'][0]   = energy['kinetic']
    energy_history['potential'][0] = energy['potential']
    energy_history['total'][0]     = energy['total']
    
    # シミュレーション実行
    print("シミュレーション計算中...")
    for i in range(1, n_steps + 1):
        energy = md.velocity_verlet_step() # 1ステップ進める

        # 結果を記録
        positions_history[i]            = md.positions.copy()
        energy_history['kinetic'][i]    = energy['kinetic']
        energy_history['potential'][i]  = energy['potential']
        energy_history['total'][i]      = energy['total']
        energy_history['time'][i]       = i * dt
    
    # エネルギープロット
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history['time'], energy_history['kinetic'],   label='Kinetic Energy')
    plt.plot(energy_history['time'], energy_history['potential'], label='Potential Energy')
    plt.plot(energy_history['time'], energy_history['total'],     label='Total Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Conservation')
    plt.legend()
    plt.grid(True)
    plt.savefig('energy_conservation.png', dpi=100)
    plt.close()
    
    # アニメーション
    if save_animation:
        create_animation(positions_history, energy_history, box_size, n_steps, n_particles)
    
    print("シミュレーション完了")
    return positions_history, energy_history


# 以下は補助関数
# アニメーション生成関数
def create_animation(positions_history, energy_history, box_size, n_steps, n_particles):
    print("アニメーション生成中...")
    
    # 表示するフレーム数を減らす (10フレームごとに1フレーム)
    skip_frames = 10
    n_frames = n_steps // skip_frames + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 粒子の色と初期位置
    colors = plt.cm.jet(np.linspace(0, 1, n_particles))
    scatter = ax1.scatter(positions_history[0, :, 0], positions_history[0, :, 1], s=80, c=colors)
    
    # 軌跡表示用の線
    trajectory_lines = [ax1.plot([], [], '-', lw=1, alpha=0.7, color=colors[i])[0] for i in range(n_particles)]
    
    # 軸設定
    ax1.set_xlim(0, box_size)
    ax1.set_ylim(0, box_size)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Particle Motion')
    ax1.grid(True)
    
    # エネルギープロット用の線
    energy_lines = [
        ax2.plot([], [], label='Kinetic Energy')[0],
        ax2.plot([], [], label='Potential Energy')[0],
        ax2.plot([], [], label='Total Energy')[0]
    ]
    
    # エネルギープロットの軸設定
    ax2.set_xlim(0, energy_history['time'][-1])
    y_min = min(min(energy_history['kinetic']), min(energy_history['potential']))
    y_max = max(max(energy_history['kinetic']), max(energy_history['total']))
    margin = (y_max - y_min) * 0.1
    ax2.set_ylim(y_min - margin, y_max + margin)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Conservation')
    ax2.legend()
    ax2.grid(True)
    
    def update(frame):
        frame_idx = min(frame * skip_frames, n_steps)
        
        # 粒子位置の更新
        scatter.set_offsets(positions_history[frame_idx])
        
        # 軌跡の更新 (最大100点まで)
        for i in range(n_particles):
            start_idx = max(0, frame_idx - 100)
            x_traj = positions_history[start_idx:frame_idx+1, i, 0]
            y_traj = positions_history[start_idx:frame_idx+1, i, 1]
            trajectory_lines[i].set_data(x_traj, y_traj)
        
        # エネルギープロットの更新
        time_data = energy_history['time'][:frame_idx+1]
        energy_lines[0].set_data(time_data, energy_history['kinetic'][:frame_idx+1])
        energy_lines[1].set_data(time_data, energy_history['potential'][:frame_idx+1])
        energy_lines[2].set_data(time_data, energy_history['total'][:frame_idx+1])
        
        return [scatter] + trajectory_lines + energy_lines
    
    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    
    print("アニメーションを保存中...")
    anim.save('md_animation.gif', writer='pillow', fps=10)
    
    plt.close()


# 後付け補助関数
# 初期位置が近く被ったことによってエネルギーが発散してしまうことを防ぐために、実装
# 重なりを持たない粒子の初期位置を生成
def generate_positions(n_particles, dim, box_size, min_dist):
    positions = []
    while len(positions) < n_particles:
        pos_candidate = np.random.uniform(0, box_size, dim)
        # 既存の位置との距離を計算
        if all(np.linalg.norm(pos_candidate - pos) >= min_dist for pos in positions):
            positions.append(pos_candidate)
    return np.array(positions)


if __name__ == "__main__":
    run_simulation(n_particles=10, n_steps=1000, dt=0.01, save_animation=True)
