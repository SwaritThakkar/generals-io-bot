/** Small seeded PRNG (mulberry32) so a room's map is reproducible from its seed. */
export class Rng {
  private s: number;

  constructor(seed: number) {
    this.s = seed >>> 0;
  }

  /** Uniform in [0, 1). */
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform integer in [lo, hi). */
  int(lo: number, hi: number): number {
    return lo + Math.floor(this.next() * (hi - lo));
  }

  /** Uniformly pick one element. */
  pick<T>(xs: readonly T[]): T {
    return xs[this.int(0, xs.length)];
  }

  /** Fisher-Yates, in place. */
  shuffle<T>(xs: T[]): T[] {
    for (let i = xs.length - 1; i > 0; i--) {
      const j = this.int(0, i + 1);
      [xs[i], xs[j]] = [xs[j], xs[i]];
    }
    return xs;
  }
}

export function randomSeed(): number {
  return (Math.random() * 0xffffffff) >>> 0;
}
