from datetime import datetime, timezone
from .io import records, text_value


def metadata(path):
    result = {}
    for row in records(path):
        title = text_value(row.get("name"))
        category = text_value(row.get("categories"))
        # Static business attributes; aggregate stars/review counts are not included.
        fine = " | ".join(f"{key}: {text_value(row[key])}"
                          for key in ("address", "city", "state", "attributes", "hours")
                          if row.get(key))
        result[str(row["business_id"])] = {"category": category, "title": title,
                                          "fine": fine or title or category}
    return result


def interactions(path):
    for ordinal, row in enumerate(records(path)):
        timestamp = int(datetime.fromisoformat(row["date"]).replace(tzinfo=timezone.utc).timestamp())
        yield {"user": str(row["user_id"]), "item": str(row["business_id"]),
               "timestamp": timestamp, "ordinal": ordinal}

