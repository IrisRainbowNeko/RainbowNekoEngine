# Tag File Format

## JSON Format Tags

In a tag file, each entry follows the format `data_path: tag`, where the tag can be any type of JSON object.
```json
{
  "data_path": "tag",
  ...
}
```

The tag can also be a dictionary, where a single image has annotations for multiple dimensions.
```json
{
  "data_path": {
    "class": "category",
    "domain": "domain"
  },
  ...
}
```

## YAML Format Tags

The YAML format tags have the same content structure as the JSON format.
```yaml
data_path: "tag"
```

```yaml
data_path:
  class: "category"
  domain: "domain"
```

## TXT Format Tags

In TXT format, each data file corresponds to its own tag file. For example, for the image `data1.png`, its annotation would be in a file named `data1.txt` located in the same folder.

```{warning}
Tags in this format can only be a string and cannot be of types like dictionaries.
```