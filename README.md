# ROOT File Explorer

## Before You Start

Remember to run

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

before we run the scripts, this makes the src folder discoverable. Ideally, put this into the profile of your terminal. Use absolute path to make terminal happy when you call the terminal at other locations.

Each folder is packaged into a python module by using an empty `__init__.py`. This makes python able to discover those modules so that we can access them.

## Understanding the Framework

Go visit the information about the [configs](./configs/info.md) involved in this framework first.
