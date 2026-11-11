"""Re-run every figure generator in the tools/fig_*.py modules."""
import sys, pathlib, importlib, inspect
sys.path.insert(0, str(pathlib.Path(__file__).parent))
made = []
for mod_name in ["fig_core", "fig_deck", "fig_ideas", "fig_castle"]:
    m = importlib.import_module(mod_name)
    fns = [(n, f) for n, f in vars(m).items()
           if inspect.isfunction(f) and f.__module__ == mod_name
           and (n.startswith("f_") or n.startswith("fig") and n[3:4].isdigit())
           and not inspect.signature(f).parameters]
    for n, f in sorted(fns):
        try:
            p = f()
            if p: made.append(pathlib.Path(p).name)
        except Exception as e:
            print(f"  !! {mod_name}.{n}: {e}")
print(f"{len(made)} svgs regenerated")
