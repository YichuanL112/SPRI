from .io import records, text_value


def metadata(path):
    result = {}
    for row in records(path):
        item = str(row["asin"])
        title = text_value(row.get("title"))
        category = text_value(row.get("categories", row.get("category")))
        fine = " | ".join(text_value(row.get(key)) for key in ("description", "brand", "price")
                          if text_value(row.get(key)))
        result[item] = {"category": category, "title": title, "fine": fine or title or category}
    return result


def interactions(path):
    for ordinal, row in enumerate(records(path)):
        yield {"user": str(row["reviewerID"]), "item": str(row["asin"]),
               "timestamp": int(row["unixReviewTime"]), "ordinal": ordinal}

