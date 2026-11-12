"""Laadi Metsaregistrist kolm 10 × 10 km õppeala (DataImport.ipynb).

Sõltuvused: pip install requests geopandas
Käivita: python scripts/get_data.py
Näide: python scripts/get_data.py --min-share 90 --output data/metsaregister

Koordinaadid on L-EST97 (EPSG:3301), järjekorras ida, põhi, meetrites.
Vaikimisi ruudud valitakse EnMAP-i polügoonide 3, 8 ja 5 keskelt.
Oma alad: --area nimi:train:600000:6460000 (ruudu edelanurk).
Kui --area on antud, asendab see kõik vaikimisi alad.
GeoJSON säilitab EPSG:3301 geomeetria ja CRS-i (GeoServeri vorming).
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import requests
import geopandas as gpd
from shapely.geometry import box
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WFS_URL = "https://gsavalik.envir.ee/geoserver/metsaregister/wfs"
CRS = "EPSG:3301"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOOTPRINTS = PROJECT_ROOT / "data/enmap_order_poly/enmap_order_poly.shp"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/metsaregister"
DEFAULT_SELECTION = [
    ("pohja_eesti", "train", 3),
    ("tartumaa", "train", 8),
    ("louna_eesti", "test", 5),
]


def load_areas(source, custom_areas=None):
    footprints = gpd.read_file(source).to_crs(CRS)
    areas = custom_areas or []
    if not custom_areas:
        for name, split, polygon_id in DEFAULT_SELECTION:
            matches = footprints.loc[footprints["id"] == polygon_id, "geometry"]
            if len(matches) != 1:
                raise ValueError(f"EnMAP-i polügooni id={polygon_id} ei leitud üheselt.")
            geometry = matches.iloc[0]
            center = geometry.centroid
            x, y = round(center.x - 5000), round(center.y - 5000)
            if not geometry.covers(box(x, y, x + 10000, y + 10000)):
                raise ValueError(f"10 km ruut ei mahu EnMAP-i polügooni {polygon_id}.")
            areas.append((name, split, x, y))
    for name, _, x, y in areas:
        square = box(x, y, x + 10000, y + 10000)
        if not any(geometry.covers(square) for geometry in footprints.geometry):
            raise ValueError(f"Ala {name} ei asu tervikuna ühe EnMAP-i polügooni sees.")
    return areas, footprints


def fetch_features(session, layer, cql_filter, page_size=1000):
    """WFS 2.0 leheküljed; lõpeta alles tühja lehe saabumisel.

    Server võib tagastada küsitud lehesuurusest vähem kirjeid.
    sys_id annab lehekülgedele stabiilse järjestuse.
    """
    start = 0
    seen = set()
    while True:
        response = session.get(WFS_URL, params={
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": f"metsaregister:{layer}",
            "outputFormat": "application/json", "srsName": CRS,
            "CQL_FILTER": cql_filter, "sortBy": "sys_id",
            "count": page_size, "startIndex": start,
        }, timeout=(15, 120))
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"WFS ei tagastanud JSON-i: {response.text[:1000]}") from exc
        if data.get("type") != "FeatureCollection":
            raise RuntimeError(f"Vigane WFS vastus: {data}")
        features = data["features"]
        if not features:
            break
        for feature in features:
            key = feature["properties"]["sys_id"]
            if key in seen:
                raise RuntimeError("WFS kordab kirjeid; allalaadimine pole täielik.")
            seen.add(key)
            yield feature
        start += len(features)


def inside_bbox(geometry, bbox):
    """Säilita ainult täielikult ruudu sees olevad polügoonid."""
    if not geometry or geometry["type"] not in ("Polygon", "MultiPolygon"):
        return False
    polygons = (geometry["coordinates"] if geometry["type"] == "MultiPolygon"
                else [geometry["coordinates"]])
    points = [point for polygon in polygons for ring in polygon for point in ring]
    xmin, ymin, xmax, ymax = bbox
    return bool(points) and all(
        xmin <= point[0] <= xmax and ymin <= point[1] <= ymax for point in points
    )


def download_area(session, area, min_age, min_share, page_size):
    name, split, x, y = area
    bbox = (x, y, x + 10000, y + 10000)
    bounds = ",".join(str(value) for value in bbox)
    cql = (f"BBOX(shape,{bounds},'{CRS}') AND keskm_vanus > {min_age} "
           "AND pindala > 0 AND peapuuliik_kood IS NOT NULL "
           "AND peapuuliik_kood <> ''")
    candidates = list(fetch_features(session, "eraldis", cql, page_size))
    stands = [f for f in candidates if inside_bbox(f["geometry"], bbox)]
    elements = defaultdict(list)
    # eraldis_element on geomeetriata: seo eraldise id, mitte sys_id abil.
    ids = sorted({int(f["properties"]["id"]) for f in stands
                  if f["properties"].get("id") is not None})
    print(f"  {len(candidates)} kandidaati, {len(stands)} täielikult ruudu sees.", flush=True)
    for offset in range(0, len(ids), 100):
        batch = ",".join(map(str, ids[offset:offset + 100]))
        query = f"eraldis_id IN ({batch}) AND rinne_kood = '1'"
        for feature in fetch_features(session, "eraldis_element", query, page_size):
            p = feature["properties"]
            elements[p["eraldis_id"]].append(p)
        print(f"  Puuliikide koosseis: {min(offset + 100, len(ids))}/{len(ids)} eraldist.", flush=True)
    selected = []
    for feature in stands:
        p = feature["properties"]
        rows = elements.get(p.get("id"), [])
        # Ära liida eri inventuuriversioone kokku.
        versions = {row.get("versioon") for row in rows}
        if len(versions) != 1:
            continue
        shares = defaultdict(float)
        valid = True
        for row in rows:
            share = row.get("osakaal")
            species = row.get("puuliik_kood")
            if not species or not isinstance(share, (int, float)) or not 0 <= share <= 100:
                valid = False
                break
            shares[species] += share
        # Puudulikku koosseisu ei tõlgendata puhta puistuna.
        if not valid or not math.isclose(sum(shares.values()), 100, abs_tol=0.01):
            continue
        share = shares.get(p["peapuuliik_kood"], 0)
        if share < min_share:
            continue
        p.update(peapuuliik_osakaal=share, study_area=name, split=split)
        selected.append(feature)
    report = {
        "name": name, "split": split, "bbox": bbox, "crs": CRS,
        "bbox_candidates": len(candidates), "fully_inside": len(stands),
        "selected": len(selected),
        "species_counts": dict(Counter(f["properties"]["peapuuliik_kood"] for f in selected)),
    }
    return selected, report


def parse_area(value):
    try:
        name, split, x, y = value.split(":")
        x, y = float(x), float(y)
        if (not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
                or split not in ("train", "test") or not all(map(math.isfinite, (x, y)))):
            raise ValueError
        return name, split, x, y
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Kasuta kuju nimi:train|test:ida:põhi") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", action="append", type=parse_area)
    parser.add_argument("--footprints", type=Path,
                        default=DEFAULT_FOOTPRINTS)
    parser.add_argument("--min-age", type=int, default=20, help="Keskmine vanus peab olema sellest suurem (20).")
    parser.add_argument("--min-share", type=float, default=80, help="Peapuuliigi minimaalne osakaal esimeses rindes, %% (80).")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    get_data(areas=args.area, footprints=args.footprints, min_age=args.min_age,
             min_share=args.min_share, page_size=args.page_size, output=args.output)


def get_data(*, areas=None, footprints=DEFAULT_FOOTPRINTS, min_age=20,
             min_share=80, page_size=1000, output=DEFAULT_OUTPUT):
    """Laadi andmed; küsi olemasolevate väljundite ülekirjutamiseks kinnitust.

    Tagastab summary.json sisu; keeldumisel tagastab None ja faile ei muuda.
    """
    source = Path(footprints)
    output = Path(output)
    if min_age < 0 or not 0 < min_share <= 100 or page_size <= 0:
        raise ValueError("Vanus peab olema >= 0, osakaal (0,100] ja lehesuurus > 0.")
    areas, footprints = load_areas(source, areas)
    if len({area[0] for area in areas}) != len(areas):
        raise ValueError("Alade nimed peavad olema erinevad.")
    for i, a in enumerate(areas):
        for b in areas[i + 1:]:
            if abs(a[2] - b[2]) < 10000 and abs(a[3] - b[3]) < 10000:
                raise ValueError("Õppealade ruudud ei tohi kattuda.")
    targets = [output / f"{area[0]}.geojson" for area in areas]
    targets += [output / "study_areas.geojson", output / "summary.json"]
    existing = [path for path in targets if path.exists()]
    if existing:
        print("Järgmised failid on juba olemas:\n" + "\n".join(str(p) for p in existing))
        try:
            answer = input("Kas soovid need andmed üle kirjutada? [jah/Ei]: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"jah", "j", "yes", "y"}:
            print("Allalaadimine tühistatud. Olemasolevaid faile ei muudetud.")
            return None
    output.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        [{"name": name, "split": split,
          "geometry": box(x, y, x + 10000, y + 10000)}
         for name, split, x, y in areas], crs=CRS,
    ).to_file(output / "study_areas.geojson", driver="GeoJSON")
    reports = []
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
        )))
        for area in areas:
            print(f"Pärin ala {area[0]} ({area[1]}) ...", flush=True)
            features, report = download_area(session, area, min_age, min_share, page_size)
            square = box(area[2], area[3], area[2] + 10000, area[3] + 10000)
            report["enmap_polygon_ids"] = [
                int(row["id"]) for _, row in footprints.iterrows()
                if row.geometry.covers(square)
            ]
            collection = {
                "type": "FeatureCollection", "name": area[0],
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3301"}},
                "features": features,
            }
            path = output / f"{area[0]}.geojson"
            path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
            reports.append(report)
            print(f"  {report['selected']} sobivat eraldist -> {path}", flush=True)
    metadata = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": WFS_URL, "min_age_exclusive": min_age,
        "enmap_footprints": str(source),
        "min_main_species_share": min_share,
        "homogeneity": "Esimese rinde peapuuliigi osakaal; koosseisu summa peab olema 100%.",
        "areas": reports,
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    main()
