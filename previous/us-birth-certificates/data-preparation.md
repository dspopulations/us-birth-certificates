> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Historical race and Hispanic-origin codes

This reference preserves source-code lists used during harmonisation. It is not
the current pipeline guide. Field availability and code meanings vary by year and
certificate revision; consult `variables.py` and the relevant NCHS user guide.

## Race source fields

Race variables include:

### `MRACE` (1989–2013, with declining revised-area use from 2003)

```
01 White
02 Black
03 American Indian / Alaskan Native
04 Chinese
05 Japanese
06 Hawaiian (includes part Hawaiian)
07 Filipino
18 Asian Indian
28 Korean
38 Samoan
48 Vietnamese
58 Guamanian
68 Other Asian / Pacific Islander in areas reporting codes 18-58
78 Combined other Asian / Pacific Islander includes 18-68 for areas that do not report them separately
```

### `MRACEREC` (2003–2013)

```
1 White
2 Black
3 American Indian / Alaskan Native
4 Asian / Pacific Islander
```

### `MBRACE` (2003–2019; list below is the later one-digit scheme)

```
1 White
2 Black
3 American Indian or Alaskan Native
4 Asian or Pacific Islander
(Puerto Rico excludes 3 and 4)
```

### `MRACE15` (from 2014)

```
01 White (only)
02 Black (only)
03 American Indian / Alaskan Native (only)
04 Asian Indian (only)
05 Chinese (only)
06 Filipino (only)
07 Japanese (only)
08 Korean (only)
09 Vietnamese (only)
10 Other Asian (only)
11 Hawaiian (only)
12 Guamanian (only)
13 Samoan (only)
14 Other Pacific Islander (only)
15 More than one race
```

### `MRACE6` (from 2018; derivable from `MRACE15`)

```
1 White (only)
2 Black (only)
3 American Indian / Alaskan Native (only)
4 Asian (only)
5 Native Hawaiian or Other Pacific Islander (only)
6 More than one race
```

## Current harmonisation

The combined race variable now has five categories, including more than one race.
Use [the current preparation guide](../../docs/data-preparation.md) for source
precedence, mappings and missing values. A present but invalid source value resolves
to NULL; it does not necessarily fall through to the next source.

Multi-race is explicit in `MRACE15` and some bridged `MBRACE` codes. Earlier
single-race fields cannot recover that detail. Do not apply `MRACE15` mappings to
`MRACE6`; the latter has only six categories.

## Hispanic-origin source fields


### `MRACEHISP` (later scheme shown)

```
1 Non-Hispanic White (only)
2 Non-Hispanic Black (only)
3 Non-Hispanic AIAN (only)
4 Non-Hispanic Asian (only)
5 Non-Hispanic NHOPI (only)
6 Non-Hispanic more than one race
7 Hispanic
8 Origin unknown or not stated
```

### `UMHISP` (2003–2013)

```
0 Non-Hispanic
1 Mexican
2 Puerto Rican
3 Cuban
4 Central American
5 Other and Unknown Hispanic
9 Origin unknown or not stated
```

### `ORRACEM` (1989–2002)

```
1 Mexican
2 Puerto Rican
3 Cuban
4 Central or South American
5 Other and unknown Hispanic
6 Non-Hispanic White
7 Non-Hispanic Black
8 Non-Hispanic other races
9 Origin unknown or not stated
```

### `MHISP_R` (from 2014)

```
0 Non-Hispanic
1 Mexican
2 Puerto Rican
3 Cuban
4 Central and South American
5 Other and Unknown Hispanic origin
9 Hispanic origin not stated
```

### `MHISPX` (from 2018)

```
0 Non-Hispanic
1 Mexican
2 Puerto Rican
3 Cuban
4 Central or South American
5 Dominican
6 Other and Unknown Hispanic
9 Origin unknown or not stated
```

## Combined origin variable

`mhisp_c` keeps Mexican, Puerto Rican and Cuban origin separately, combines other
Hispanic origins, and distinguishes non-Hispanic from unknown origin. The current
SQL uses `MHISP_R`, then `MHISPX`, `UMHISP` and `ORRACEM` when earlier sources are
absent. See the [current mappings](../../docs/data-preparation.md).

The combined race/origin variable is reconstructed from harmonised inputs. Raw
`MRACEHISP` codes have different meanings across years; the list above describes
the later scheme only. Check the source guide for each year's exact categories.
