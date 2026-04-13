from __future__ import annotations

import winreg


def iter_uninstall_entries(root, path):
    try:
        with winreg.OpenKey(root, path) as k:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(k, i)
                except OSError:
                    break
                i += 1
                try:
                    with winreg.OpenKey(k, sub) as sk:
                        name, _ = winreg.QueryValueEx(sk, "DisplayName")
                        version = None
                        try:
                            version, _ = winreg.QueryValueEx(sk, "DisplayVersion")
                        except OSError:
                            pass
                        yield str(name), (str(version) if version is not None else None)
                except OSError:
                    continue
    except OSError:
        return


def main() -> None:
    targets = []
    needles = ("Microsoft Visual C++", "2015", "2022")

    locations = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    for root, path in locations:
        for name, version in iter_uninstall_entries(root, path):
            low = name.lower()
            if all(n.lower() in low for n in needles):
                targets.append((name, version))

    if not targets:
        print("VC++ 2015-2022 runtime: NOT FOUND in uninstall registry.")
        return

    print("VC++ 2015-2022 runtime entries found:")
    for name, version in sorted(targets):
        print(f"- {name}  (version={version})")


if __name__ == "__main__":
    main()

