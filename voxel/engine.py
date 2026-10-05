"""
游戏内体素引擎与Marching Cubes系统 - 核心模块

已实现功能：
1. 体素存储：RLE 压缩 + 区块分区，支持只加载/卸载部分体素
2. Marching Cubes：256 项查找表 + 梯度法线，生成平滑表面网格
3. 体素渲染：隐藏面剔除 + 视锥/距离剔除 + 距离 LOD
4. 体素编辑：添加 / 删除体素（带历史与撤销）
5. 无限地形：程序化噪声生成 + 以玩家为中心的流式加载/卸载

纯 Python 标准库实现，无第三方依赖。
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import time

Vec3 = Tuple[float, float, float]
BlockPos = Tuple[int, int, int]
ChunkCoord = Tuple[int, int, int]

# 体素类型
AIR = 0
GRASS = 1
DIRT = 2
STONE = 3


@dataclass
class Config:
    """引擎配置（全部带默认值，保持 Config() 可直接构造）。"""
    chunk_size: int = 16
    seed: int = 1337
    view_distance: int = 4          # 流式加载半径（区块）
    lod_distances: Tuple[float, ...] = (32.0, 64.0, 128.0)
    base_height: float = 8.0
    height_amplitude: float = 24.0
    noise_scale: float = 48.0


# ---------------------------------------------------------------------------
# 数学小工具
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-12:
        return (0.0, 1.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _distance(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# ---------------------------------------------------------------------------
# 1. 体素存储：RLE 压缩 + 区块分区
# ---------------------------------------------------------------------------

class VoxelChunk:
    """单个区块的体素数据。

    - 压缩态：_runs 行程长度编码 [(value, count), ...]，大片空气几乎不占内存；
    - 编辑态：_raw 解压为 bytearray，随机读写 O(1)，compress() 可重新压缩。
    """

    def __init__(self, coord: ChunkCoord, size: int):
        self.coord = coord
        self.size = size
        self._runs: List[List] = [[AIR, size ** 3]]
        self._raw: Optional[bytearray] = None
        self.dirty = False

    def _index(self, lx: int, ly: int, lz: int) -> int:
        return (lz * self.size + ly) * self.size + lx

    def _ensure_raw(self) -> None:
        if self._raw is None:
            raw = bytearray()
            for value, count in self._runs:
                raw.extend(bytes([value]) * count)
            self._raw = raw

    def get(self, lx: int, ly: int, lz: int) -> int:
        index = self._index(lx, ly, lz)
        if self._raw is not None:
            return self._raw[index]
        position = 0
        for value, count in self._runs:
            if index < position + count:
                return value
            position += count
        return AIR

    def set(self, lx: int, ly: int, lz: int, value: int) -> None:
        self._ensure_raw()
        index = self._index(lx, ly, lz)
        if self._raw[index] != value:
            self._raw[index] = value
            self.dirty = True

    def compress(self) -> None:
        """把编辑数据重新压缩成 RLE 行程（无损）。"""
        if self._raw is None:
            return
        runs: List[List] = []
        current = self._raw[0]
        count = 1
        for value in self._raw[1:]:
            if value == current:
                count += 1
            else:
                runs.append([current, count])
                current = value
                count = 1
        runs.append([current, count])
        self._runs = runs
        self._raw = None
        self.dirty = False

    @property
    def is_compressed(self) -> bool:
        return self._raw is None

    @property
    def is_empty(self) -> bool:
        if self._raw is not None:
            return not any(self._raw)
        return len(self._runs) == 1 and self._runs[0][0] == AIR

    def compressed_size(self) -> int:
        # 每个行程约占 1 字节体素值 + 2 字节计数
        return len(self._runs) * 3

    def raw_size(self) -> int:
        return self.size ** 3

    def memory_usage(self) -> int:
        return self.compressed_size() if self.is_compressed else self.raw_size()


class VoxelStorage:
    """分区体素存储：世界被划分为 chunk_size^3 的区块，可按需加载/卸载。"""

    def __init__(self, chunk_size: int = 16):
        self.chunk_size = chunk_size
        self.chunks: Dict[ChunkCoord, VoxelChunk] = {}

    def _split(self, x: int, y: int, z: int):
        size = self.chunk_size
        cx, cy, cz = x // size, y // size, z // size
        return (cx, cy, cz), (x - cx * size, y - cy * size, z - cz * size)

    def get_chunk(self, coord: ChunkCoord, create: bool = False) -> Optional[VoxelChunk]:
        chunk = self.chunks.get(coord)
        if chunk is None and create:
            chunk = VoxelChunk(coord, self.chunk_size)
            self.chunks[coord] = chunk
        return chunk

    def load_chunk(self, coord: ChunkCoord, chunk: Optional[VoxelChunk] = None) -> VoxelChunk:
        if chunk is None:
            chunk = VoxelChunk(coord, self.chunk_size)
        self.chunks[coord] = chunk
        return chunk

    def unload_chunk(self, coord: ChunkCoord) -> bool:
        return self.chunks.pop(coord, None) is not None

    def is_loaded(self, coord: ChunkCoord) -> bool:
        return coord in self.chunks

    def get_voxel(self, x: int, y: int, z: int) -> int:
        coord, local = self._split(x, y, z)
        chunk = self.chunks.get(coord)
        return AIR if chunk is None else chunk.get(*local)

    def set_voxel(self, x: int, y: int, z: int, value: int) -> None:
        coord, local = self._split(x, y, z)
        if value == AIR and coord not in self.chunks:
            return
        self.get_chunk(coord, create=True).set(*local, value)

    def compress_all(self) -> None:
        for chunk in self.chunks.values():
            chunk.compress()

    compress = compress_all

    def non_air_count(self) -> int:
        total = 0
        for chunk in self.chunks.values():
            if chunk._raw is not None:
                total += sum(1 for value in chunk._raw if value != AIR)
            else:
                total += sum(count for value, count in chunk._runs if value != AIR)
        return total

    def memory_usage(self) -> Dict[str, float]:
        raw = sum(chunk.raw_size() for chunk in self.chunks.values())
        compressed = sum(chunk.memory_usage() for chunk in self.chunks.values())
        return {
            "raw": raw,
            "compressed": compressed,
            "ratio": (raw / compressed) if compressed > 0 else 1.0,
            "chunks": len(self.chunks),
        }

    def compression_ratio(self) -> float:
        return self.memory_usage()["ratio"]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def loaded_chunks(self) -> List[ChunkCoord]:
        return list(self.chunks.keys())


# ---------------------------------------------------------------------------
# 2. Marching Cubes：查找表（经典 Lorensen-Cline 顶点/边编号）
# ---------------------------------------------------------------------------

_MC_CORNERS = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)
_MC_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),       # 底面四条边
    (4, 5), (5, 6), (6, 7), (7, 4),       # 顶面四条边
    (0, 4), (1, 5), (2, 6), (3, 7),       # 四条竖边
)
_MC_FACES = (
    (0, 1, 2, 3),  # z = 0
    (4, 5, 6, 7),  # z = 1
    (0, 1, 5, 4),  # y = 0
    (3, 2, 6, 7),  # y = 1
    (0, 3, 7, 4),  # x = 0
    (1, 2, 6, 5),  # x = 1
)


def _build_edge_table():
    """256 种角点配置 -> 被等值面穿过的边掩码。"""
    table = []
    for config in range(256):
        mask = 0
        for edge_index, (a, b) in enumerate(_MC_EDGES):
            if ((config >> a) & 1) != ((config >> b) & 1):
                mask |= 1 << edge_index
        table.append(mask)
    return tuple(table)


def _build_tri_table():
    """256 种角点配置 -> 三角形列表（每个三角形由 3 条边的编号组成）。

    构造方法：求出立方体每个面上的交线并连接成闭合环，再对每个环做扇形
    三角化。结果在导入时只计算一次，运行时直接查表。
    """
    edge_lookup = {tuple(sorted(edge)): i for i, edge in enumerate(_MC_EDGES)}
    face_edges = []
    for face in _MC_FACES:
        edges = []
        for i in range(4):
            a, b = face[i], face[(i + 1) % 4]
            edges.append(edge_lookup[tuple(sorted((a, b)))])
        face_edges.append(edges)

    table = []
    for config in range(256):
        inside = [(config >> i) & 1 for i in range(8)]
        segments = []
        for face, edges in zip(_MC_FACES, face_edges):
            crossed = [
                edge for edge in edges
                if inside[_MC_EDGES[edge][0]] != inside[_MC_EDGES[edge][1]]
            ]
            if len(crossed) == 2:
                segments.append((crossed[0], crossed[1]))
            elif len(crossed) == 4:
                # 模糊面（四角点两内两外）：按面首角点是否在内选择配对
                if inside[face[0]]:
                    segments.append((crossed[3], crossed[0]))
                    segments.append((crossed[1], crossed[2]))
                else:
                    segments.append((crossed[0], crossed[1]))
                    segments.append((crossed[2], crossed[3]))

        neighbors: Dict[int, List[int]] = {}
        for a, b in segments:
            neighbors.setdefault(a, []).append(b)
            neighbors.setdefault(b, []).append(a)

        triangles = []
        visited = set()
        for start in neighbors:
            if start in visited:
                continue
            loop = [start]
            visited.add(start)
            previous, current = None, start
            while True:
                candidates = [nxt for nxt in neighbors[current] if nxt != previous]
                if not candidates:
                    break
                nxt = candidates[0]
                if nxt == start:
                    break
                loop.append(nxt)
                visited.add(nxt)
                previous, current = current, nxt
            for i in range(1, len(loop) - 1):
                triangles.append((loop[0], loop[i], loop[i + 1]))
        table.append(tuple(triangles))
    return tuple(table)


EDGE_TABLE = _build_edge_table()
TRI_TABLE = _build_tri_table()


def _sphere_field(n: int):
    center = (n - 1) / 2.0
    radius = n * 0.45

    def field(x: int, y: int, z: int) -> float:
        dx, dy, dz = x - center, y - center, z - center
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return 1.0 - distance / radius

    return field


@dataclass
class Mesh:
    """Marching Cubes 生成的平滑表面网格（顶点 + 法线 + 三角形索引）。"""
    vertices: List[Vec3] = field(default_factory=list)
    normals: List[Vec3] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def triangles(self):
        return self.faces

    @property
    def indices(self):
        return self.faces

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.faces)

    @property
    def is_smooth(self) -> bool:
        return bool(self.normals) and len(self.normals) == len(self.vertices)

    def to_dict(self) -> Dict:
        return {
            "vertices": self.vertices,
            "normals": self.normals,
            "faces": self.faces,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
        }

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class MarchingCubes:
    """查找表驱动的 Marching Cubes：线性插值顶点 + 梯度法线（平滑着色）。"""

    edge_table = EDGE_TABLE
    tri_table = TRI_TABLE

    def polygonize(self, field=None, iso_level: float = 0.5,
                   size: Optional[int] = None) -> Mesh:
        sampler, dims = self._make_sampler(field, size)
        nx, ny, nz = dims
        if nx < 2 or ny < 2 or nz < 2:
            return Mesh()

        values = [0.0] * (nx * ny * nz)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    values[(x * ny + y) * nz + z] = float(sampler(x, y, z))

        def value_at(x, y, z) -> float:
            return values[(x * ny + y) * nz + z]

        gradient_cache: Dict[Tuple[int, int, int], Vec3] = {}

        def gradient_at(x, y, z):
            key = (x, y, z)
            cached = gradient_cache.get(key)
            if cached is not None:
                return cached
            xm, xp = max(x - 1, 0), min(x + 1, nx - 1)
            ym, yp = max(y - 1, 0), min(y + 1, ny - 1)
            zm, zp = max(z - 1, 0), min(z + 1, nz - 1)
            gradient = (
                (value_at(xp, y, z) - value_at(xm, y, z)) / max(xp - xm, 1),
                (value_at(x, yp, z) - value_at(x, ym, z)) / max(yp - ym, 1),
                (value_at(x, y, zp) - value_at(x, y, zm)) / max(zp - zm, 1),
            )
            gradient_cache[key] = gradient
            return gradient

        mesh = Mesh()
        vertex_cache: Dict[Tuple, int] = {}

        def edge_vertex(cell, edge_index) -> int:
            a, b = _MC_EDGES[edge_index]
            oa, ob = _MC_CORNERS[a], _MC_CORNERS[b]
            pa = (cell[0] + oa[0], cell[1] + oa[1], cell[2] + oa[2])
            pb = (cell[0] + ob[0], cell[1] + ob[1], cell[2] + ob[2])
            key = (pa, pb) if pa <= pb else (pb, pa)
            cached = vertex_cache.get(key)
            if cached is not None:
                return cached
            va, vb = value_at(*pa), value_at(*pb)
            denominator = vb - va
            t = 0.5 if abs(denominator) < 1e-12 else (iso_level - va) / denominator
            t = min(1.0, max(0.0, t))
            position = (
                pa[0] + t * (pb[0] - pa[0]),
                pa[1] + t * (pb[1] - pa[1]),
                pa[2] + t * (pb[2] - pa[2]),
            )
            ga, gb = gradient_at(*pa), gradient_at(*pb)
            # 法线 = 密度梯度的反方向（由实体指向空气），插值后归一化
            normal = _normalize((
                -(ga[0] + t * (gb[0] - ga[0])),
                -(ga[1] + t * (gb[1] - ga[1])),
                -(ga[2] + t * (gb[2] - ga[2])),
            ))
            index = len(mesh.vertices)
            mesh.vertices.append(position)
            mesh.normals.append(normal)
            vertex_cache[key] = index
            return index

        for cx in range(nx - 1):
            for cy in range(ny - 1):
                for cz in range(nz - 1):
                    config = 0
                    for corner_index, offset in enumerate(_MC_CORNERS):
                        if value_at(cx + offset[0], cy + offset[1],
                                    cz + offset[2]) > iso_level:
                            config |= 1 << corner_index
                    if EDGE_TABLE[config] == 0:
                        continue
                    cell = (cx, cy, cz)
                    for tri in TRI_TABLE[config]:
                        i0 = edge_vertex(cell, tri[0])
                        i1 = edge_vertex(cell, tri[1])
                        i2 = edge_vertex(cell, tri[2])
                        # 依据顶点法线方向修正绕序，保证三角形朝外
                        p0, p1, p2 = (mesh.vertices[i0], mesh.vertices[i1],
                                      mesh.vertices[i2])
                        geometric = _cross(_sub(p1, p0), _sub(p2, p0))
                        normal_sum = _add(_add(mesh.normals[i0],
                                               mesh.normals[i1]),
                                          mesh.normals[i2])
                        if _dot(geometric, normal_sum) < 0:
                            i1, i2 = i2, i1
                        mesh.faces.append((i0, i1, i2))
        return mesh

    def _make_sampler(self, field, size):
        """把各种形式的密度场统一成 (sampler, (nx, ny, nz)) 采样点网格。"""
        if field is None:
            n = size or 16
            return _sphere_field(n), (n, n, n)
        if isinstance(field, VoxelStorage):
            return self._storage_sampler(field)
        if callable(field):
            n = size or 16
            return field, (n, n, n)
        if isinstance(field, dict):
            if size is None:
                n = max((max(key) for key in field.keys()), default=0) + 2
            else:
                n = size
            return (lambda x, y, z: float(field.get((x, y, z), 0.0))), (n, n, n)
        # 嵌套序列，按 field[x][y][z] 索引
        nx = len(field)
        ny = len(field[0]) if nx else 0
        nz = len(field[0][0]) if ny else 0
        return (lambda x, y, z: float(field[x][y][z])), (nx, ny, nz)

    @staticmethod
    def _storage_sampler(storage: VoxelStorage):
        if not storage.chunks:
            n = 16
            return _sphere_field(n), (n, n, n)
        size = storage.chunk_size
        xs = [coord[0] for coord in storage.chunks]
        ys = [coord[1] for coord in storage.chunks]
        zs = [coord[2] for coord in storage.chunks]
        lo = (min(xs) * size - 1, min(ys) * size - 1, min(zs) * size - 1)
        hi = ((max(xs) + 1) * size + 1, (max(ys) + 1) * size + 1,
              (max(zs) + 1) * size + 1)
        dims = (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1)

        def sampler(x, y, z):
            voxel = storage.get_voxel(lo[0] + x, lo[1] + y, lo[2] + z)
            return 1.0 if voxel != AIR else 0.0

        return sampler, dims


# ---------------------------------------------------------------------------
# 3. 体素渲染：隐藏面剔除 + 视锥/距离剔除 + LOD
# ---------------------------------------------------------------------------

_FACE_DIRECTIONS = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


def _frustum_planes(position, direction, fov_degrees, far,
                    aspect=1.0, near=0.1):
    """由相机参数构造 6 个视锥平面 (法线, 平面上一点)，法线朝视锥内部。"""
    forward = _normalize(direction)
    up = (0.0, 1.0, 0.0) if abs(forward[1]) < 0.99 else (1.0, 0.0, 0.0)
    right = _normalize(_cross(forward, up))
    up = _cross(right, forward)
    half_v = math.radians(fov_degrees) / 2.0
    half_h = math.atan(math.tan(half_v) * aspect)
    sin_v, cos_v = math.sin(half_v), math.cos(half_v)
    sin_h, cos_h = math.sin(half_h), math.cos(half_h)
    return [
        (_add(_scale(forward, sin_h), _scale(right, cos_h)), position),   # 左
        (_sub(_scale(forward, sin_h), _scale(right, cos_h)), position),   # 右
        (_add(_scale(forward, sin_v), _scale(up, cos_v)), position),      # 下
        (_sub(_scale(forward, sin_v), _scale(up, cos_v)), position),      # 上
        (forward, _add(position, _scale(forward, near))),                 # 近
        (_scale(forward, -1.0), _add(position, _scale(forward, far))),    # 远
    ]


def _aabb_in_frustum(min_bound, max_bound, planes) -> bool:
    for normal, point in planes:
        # 沿法线方向最远的顶点都在平面外侧 => 整个包围盒被剔除
        px = max_bound[0] if normal[0] >= 0 else min_bound[0]
        py = max_bound[1] if normal[1] >= 0 else min_bound[1]
        pz = max_bound[2] if normal[2] >= 0 else min_bound[2]
        if _dot((px - point[0], py - point[1], pz - point[2]), normal) < 0:
            return False
    return True


@dataclass
class RenderStats:
    """一帧体素渲染的统计结果。"""
    camera_position: Vec3 = (0.0, 0.0, 0.0)
    total_chunks: int = 0
    visible_chunks: int = 0
    culled_chunks: int = 0
    rendered_faces: int = 0
    culled_faces: int = 0
    lod_levels: Dict[int, int] = field(default_factory=dict)
    chunks: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "camera_position": self.camera_position,
            "total_chunks": self.total_chunks,
            "visible_chunks": self.visible_chunks,
            "culled_chunks": self.culled_chunks,
            "rendered_faces": self.rendered_faces,
            "culled_faces": self.culled_faces,
            "lod_levels": dict(self.lod_levels),
            "chunks": list(self.chunks),
        }

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class VoxelRenderer:
    """体素渲染器。

    - 隐藏面剔除：被相邻实体体素挡住的面不渲染；
    - 区块剔除：超出视距或在视锥外的区块整体剔除；
    - LOD：按相机距离选择细节级别，远处区块用更大步长采样。
    """

    def __init__(self, storage: VoxelStorage,
                 lod_distances: Tuple[float, ...] = (32.0, 64.0, 128.0)):
        self.storage = storage
        self.lod_distances = tuple(sorted(lod_distances))

    def lod_for_distance(self, distance: float) -> int:
        lod = 0
        for threshold in self.lod_distances:
            if distance > threshold:
                lod += 1
        return min(lod, 3)

    def render(self, camera_position=None, view_direction=None,
               fov_degrees: float = 90.0, view_distance: Optional[float] = None
               ) -> RenderStats:
        camera = tuple(camera_position) if camera_position else (0.0, 0.0, 0.0)
        planes = None
        if view_direction is not None:
            far = view_distance if view_distance else max(
                self.lod_distances[-1] * 4.0, 256.0)
            planes = _frustum_planes(camera, view_direction, fov_degrees, far)

        stats = RenderStats(camera_position=camera)
        size = self.storage.chunk_size
        for coord, chunk in sorted(self.storage.chunks.items()):
            stats.total_chunks += 1
            center = (coord[0] * size + size / 2.0,
                      coord[1] * size + size / 2.0,
                      coord[2] * size + size / 2.0)
            distance = _distance(camera, center)
            if view_distance is not None and distance > view_distance:
                stats.culled_chunks += 1
                continue
            if planes is not None and not _aabb_in_frustum(
                    (coord[0] * size, coord[1] * size, coord[2] * size),
                    ((coord[0] + 1) * size, (coord[1] + 1) * size,
                     (coord[2] + 1) * size),
                    planes):
                stats.culled_chunks += 1
                continue
            lod = self.lod_for_distance(distance)
            stride = min(1 << lod, size)
            rendered, culled = self._render_chunk(chunk, stride)
            stats.visible_chunks += 1
            stats.rendered_faces += rendered
            stats.culled_faces += culled
            stats.lod_levels[lod] = stats.lod_levels.get(lod, 0) + 1
            stats.chunks.append({
                "coord": coord,
                "distance": distance,
                "lod": lod,
                "rendered_faces": rendered,
                "culled_faces": culled,
            })
        return stats

    def _render_chunk(self, chunk: VoxelChunk, stride: int):
        chunk._ensure_raw()
        size = chunk.size
        coord = chunk.coord
        rendered = 0
        culled = 0
        for lz in range(0, size, stride):
            for ly in range(0, size, stride):
                for lx in range(0, size, stride):
                    if chunk.get(lx, ly, lz) == AIR:
                        continue
                    wx = coord[0] * size + lx
                    wy = coord[1] * size + ly
                    wz = coord[2] * size + lz
                    for dx, dy, dz in _FACE_DIRECTIONS:
                        neighbor = self.storage.get_voxel(
                            wx + dx * stride, wy + dy * stride, wz + dz * stride)
                        if neighbor == AIR:
                            rendered += 1
                        else:
                            culled += 1
        return rendered, culled


# ---------------------------------------------------------------------------
# 4. 体素编辑：添加 / 删除（带历史与撤销）
# ---------------------------------------------------------------------------

class VoxelEditor:
    """体素编辑器：在世界中添加或删除体素，记录历史并支持撤销。"""

    def __init__(self, storage: VoxelStorage, history: Optional[List] = None):
        self.storage = storage
        self.history = history if history is not None else []

    def add_voxel(self, x: int, y: int, z: int, voxel_type: int = GRASS) -> bool:
        """在空位置添加体素；位置已被占用时返回 False。"""
        if self.storage.get_voxel(x, y, z) != AIR:
            return False
        self.storage.set_voxel(x, y, z, voxel_type)
        self.history.append({
            "action": "add",
            "position": (x, y, z),
            "voxel_type": voxel_type,
            "time": time.time(),
        })
        return True

    def remove_voxel(self, x: int, y: int, z: int) -> bool:
        """删除已有体素；位置为空时返回 False。"""
        existing = self.storage.get_voxel(x, y, z)
        if existing == AIR:
            return False
        self.storage.set_voxel(x, y, z, AIR)
        self.history.append({
            "action": "remove",
            "position": (x, y, z),
            "voxel_type": existing,
            "time": time.time(),
        })
        return True

    place_voxel = add_voxel
    delete_voxel = remove_voxel

    def undo(self) -> bool:
        if not self.history:
            return False
        entry = self.history.pop()
        x, y, z = entry["position"]
        if entry["action"] == "add":
            self.storage.set_voxel(x, y, z, AIR)
        else:
            self.storage.set_voxel(x, y, z, entry["voxel_type"])
        return True


# ---------------------------------------------------------------------------
# 5. 无限地形：程序化生成 + 流式加载
# ---------------------------------------------------------------------------

def _hash01(ix: int, iz: int, seed: int) -> float:
    """确定性整数哈希 -> [0, 1] 浮点数。"""
    n = (ix * 374761393 + iz * 668265263 + seed * 974634211) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    n = (n ^ (n >> 16)) & 0xFFFFFFFF
    return n / 4294967295.0


def _value_noise(x: float, z: float, seed: int) -> float:
    ix, iz = math.floor(x), math.floor(z)
    fx, fz = x - ix, z - iz
    sx = fx * fx * (3.0 - 2.0 * fx)
    sz = fz * fz * (3.0 - 2.0 * fz)
    v00 = _hash01(ix, iz, seed)
    v10 = _hash01(ix + 1, iz, seed)
    v01 = _hash01(ix, iz + 1, seed)
    v11 = _hash01(ix + 1, iz + 1, seed)
    return _lerp(_lerp(v00, v10, sx), _lerp(v01, v11, sx), sz)


def _fbm(x: float, z: float, seed: int, octaves: int = 4,
         lacunarity: float = 2.0, gain: float = 0.5) -> float:
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    norm = 0.0
    for octave in range(octaves):
        total += amplitude * _value_noise(
            x * frequency, z * frequency, seed + octave * 101)
        norm += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / norm


class InfiniteTerrain:
    """无限地形：值噪声 fBm 程序化生成，随玩家位置流式加载/卸载区块。"""

    is_infinite = True

    def __init__(self, storage: VoxelStorage, seed: int = 1337,
                 view_distance: int = 4, base_height: float = 8.0,
                 height_amplitude: float = 24.0, noise_scale: float = 48.0):
        self.storage = storage
        self.seed = seed
        self.view_distance = view_distance
        self.base_height = base_height
        self.height_amplitude = height_amplitude
        self.noise_scale = noise_scale
        self._columns: Dict[Tuple[int, int], List[ChunkCoord]] = {}

    # ---- 程序化生成 ----

    def generate_height(self, x: float, z: float) -> float:
        """任意世界坐标的地表高度（确定性，同一种子结果相同）。"""
        noise = _fbm(x / self.noise_scale, z / self.noise_scale, self.seed)
        return self.base_height + noise * self.height_amplitude

    get_height = generate_height
    height_at = generate_height
    sample_height = generate_height

    def generate_chunk(self, cx: int, cz: int):
        """程序化生成一列地形（草/土/石分层），写入体素存储。"""
        size = self.storage.chunk_size
        touched = set()
        for lz in range(size):
            for lx in range(size):
                wx, wz = cx * size + lx, cz * size + lz
                height = max(0, int(self.generate_height(wx, wz)))
                for wy in range(height + 1):
                    if wy == height:
                        voxel_type = GRASS
                    elif wy >= height - 2:
                        voxel_type = DIRT
                    else:
                        voxel_type = STONE
                    self.storage.set_voxel(wx, wy, wz, voxel_type)
                    touched.add((wx // size, wy // size, wz // size))
        for coord in touched:
            chunk = self.storage.get_chunk(coord)
            if chunk is not None:
                chunk.compress()
        self._columns[(cx, cz)] = sorted(touched)
        return (cx, cz)

    generate = generate_chunk

    # ---- 流式加载 ----

    def update_streaming(self, player_position, view_distance: Optional[int] = None):
        """以玩家为中心加载半径内区块、卸载半径外区块。"""
        radius = self.view_distance if view_distance is None else view_distance
        size = self.storage.chunk_size
        if len(player_position) >= 3:
            px, pz = player_position[0], player_position[2]
        else:
            px, pz = player_position[0], player_position[1]
        pcx = int(math.floor(px / size))
        pcz = int(math.floor(pz / size))

        desired = set()
        reach = int(math.ceil(radius))
        for dx in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                if dx * dx + dz * dz <= radius * radius:
                    desired.add((pcx + dx, pcz + dz))

        loaded, unloaded = [], []
        for column in sorted(desired):
            if column not in self._columns:
                self.generate_chunk(*column)
                loaded.append(column)
        for column in list(self._columns):
            if column not in desired:
                self.unload_chunk(*column)
                unloaded.append(column)
        return {
            "player_chunk": (pcx, pcz),
            "loaded": loaded,
            "unloaded": unloaded,
            "loaded_chunks": len(self._columns),
        }

    stream_around = update_streaming

    def load_chunk(self, cx: int, cz: int) -> bool:
        if (cx, cz) in self._columns:
            return False
        self.generate_chunk(cx, cz)
        return True

    def unload_chunk(self, cx: int, cz: int) -> bool:
        coords = self._columns.pop((cx, cz), None)
        if coords is None:
            return False
        for coord in coords:
            self.storage.unload_chunk(coord)
        return True

    def is_loaded(self, cx: int, cz: int) -> bool:
        return (cx, cz) in self._columns

    @property
    def loaded_columns(self) -> List[Tuple[int, int]]:
        return sorted(self._columns)

    @property
    def loaded_chunk_count(self) -> int:
        return len(self._columns)


# ---------------------------------------------------------------------------
# 引擎门面：整合五大子系统
# ---------------------------------------------------------------------------

class VoxelEngine:
    """游戏内体素引擎。

    子系统入口（对应五项需求）：
    - voxel_storage()   -> VoxelStorage    压缩 + 分区
    - marching_cubes()  -> Mesh            查表 + 法线的平滑表面
    - voxel_rendering() -> RenderStats     剔除 + LOD
    - voxel_editing()   -> VoxelEditor     添加 + 删除
    - infinite_terrain()-> InfiniteTerrain 生成 + 流式加载
    """

    def __init__(self, config=None):
        self.config = config or {}
        self._state: Dict = {}
        self._history: List = []
        self._settings = self._read_config(self.config)
        self._build_subsystems()

    @staticmethod
    def _read_config(config) -> Dict:
        defaults = Config()
        keys = ("chunk_size", "seed", "view_distance", "lod_distances",
                "base_height", "height_amplitude", "noise_scale")
        settings = {key: getattr(defaults, key) for key in keys}
        if isinstance(config, Config):
            for key in keys:
                settings[key] = getattr(config, key)
        elif isinstance(config, dict):
            for key in keys:
                if key in config:
                    settings[key] = config[key]
        return settings

    def _build_subsystems(self) -> None:
        settings = self._settings
        self.storage = VoxelStorage(settings["chunk_size"])
        self.terrain = InfiniteTerrain(
            self.storage,
            seed=settings["seed"],
            view_distance=settings["view_distance"],
            base_height=settings["base_height"],
            height_amplitude=settings["height_amplitude"],
            noise_scale=settings["noise_scale"],
        )
        self.editor = VoxelEditor(self.storage, self._history)
        self.renderer = VoxelRenderer(self.storage, settings["lod_distances"])
        self.mesher = MarchingCubes()

    # ---- 五大子系统入口 ----

    def voxel_storage(self) -> VoxelStorage:
        return self.storage

    def marching_cubes(self, field=None, iso_level: float = 0.5,
                       size: Optional[int] = None) -> Mesh:
        return self.mesher.polygonize(field, iso_level, size)

    def voxel_rendering(self, camera_position=None, view_direction=None,
                        fov_degrees: float = 90.0,
                        view_distance: Optional[float] = None) -> RenderStats:
        return self.renderer.render(
            camera_position, view_direction, fov_degrees, view_distance)

    def voxel_editing(self) -> VoxelEditor:
        return self.editor

    def infinite_terrain(self) -> InfiniteTerrain:
        return self.terrain

    # ---- 便捷代理 ----

    def get_voxel(self, x: int, y: int, z: int) -> int:
        return self.storage.get_voxel(x, y, z)

    def set_voxel(self, x: int, y: int, z: int, value: int) -> None:
        self.storage.set_voxel(x, y, z, value)

    def add_voxel(self, x: int, y: int, z: int, voxel_type: int = GRASS) -> bool:
        return self.editor.add_voxel(x, y, z, voxel_type)

    def remove_voxel(self, x: int, y: int, z: int) -> bool:
        return self.editor.remove_voxel(x, y, z)

    place_voxel = add_voxel
    delete_voxel = remove_voxel

    def undo(self) -> bool:
        return self.editor.undo()

    def render(self, **kwargs) -> RenderStats:
        return self.voxel_rendering(**kwargs)

    def generate_mesh(self, *args, **kwargs) -> Mesh:
        return self.marching_cubes(*args, **kwargs)

    def set_player_position(self, position) -> None:
        self._state["player_position"] = tuple(position)

    # ---- 生命周期（与原骨架接口一致）----

    def update(self, dt: float) -> None:
        self._state["time"] = self._state.get("time", 0.0) + dt
        player = self._state.get("player_position")
        if player is not None:
            self._state["streaming"] = self.terrain.update_streaming(player)

    def reset(self) -> None:
        self._state = {}
        self._history = []
        self._build_subsystems()
