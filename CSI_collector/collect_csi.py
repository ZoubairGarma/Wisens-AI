"""
collect_csi.py - Capture des mesures CSI WISENS-AI depuis le port serie
                 et sauvegarde dans un fichier CSV nomme par experience,
                 avec les metadonnees d'experience en colonnes.

                 La capture s'arrete automatiquement apres une duree fixe
                 (5 minutes par defaut), pour garantir des sessions de
                 longueur homogene entre scenarios (voir dossier projet,
                 section 5.3 "Contraintes d'installation").

Usage:
    python collect_csi.py EXP_001 piece_vide --distance 3.0 --comment "test initial"
    python collect_csi.py EXP_002 mouvement_faible --ground-truth presence_mouvement
    python collect_csi.py EXP_003 mouvement_fort --duration 180
"""

import argparse
import csv
import os
import sys
import time

import serial

PORT = "COM3"
BAUD = 115200
OUTPUT_DIR = "data"
DEFAULT_DURATION_S = 300 # 5 minutes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture des mesures CSI WISENS-AI vers un fichier CSV, "
                     "pendant une duree fixe."
    )
    parser.add_argument("experiment_id", help="Identifiant de l'experience, ex: EXP_001")
    parser.add_argument("scenario", help="Nom du scenario, ex: piece_vide, mouvement_faible")
    parser.add_argument("--zone", default="ZONE_01", help="Identifiant de la zone (defaut: ZONE_01)")
    parser.add_argument("--distance", type=float, default=0.0,
                         help="Distance emetteur/recepteur en metres (defaut: 0.0)")
    parser.add_argument("--ground-truth", default=None,
                         help="Etat reel observe (defaut: identique au scenario)")
    parser.add_argument("--comment", default="",
                         help="Commentaire libre sur les conditions de test")
    parser.add_argument("--port", default=PORT, help=f"Port serie (defaut: {PORT})")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION_S,
                         help=f"Duree de capture en secondes (defaut: {DEFAULT_DURATION_S} "
                              f"= 5 minutes). La capture s'arrete automatiquement "
                              f"a l'ecoulement de ce delai.")
    return parser.parse_args()


def build_output_path(experiment_id: str, scenario: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{experiment_id}_{scenario}.csv"
    return os.path.join(OUTPUT_DIR, filename)


def format_remaining(seconds_left: float) -> str:
    minutes, seconds = divmod(int(max(seconds_left, 0)), 60)
    return f"{minutes:02d}:{seconds:02d}"


def main():
    args = parse_args()
    ground_truth = args.ground_truth if args.ground_truth else args.scenario
    output_path = build_output_path(args.experiment_id, args.scenario)

    if os.path.exists(output_path):
        print(f"ATTENTION: {output_path} existe deja et sera ECRASE.")
        confirm = input("Continuer ? (o/n) : ").strip().lower()
        if confirm != "o":
            print("Annule.")
            sys.exit(0)

    try:
        ser = serial.Serial(args.port, BAUD, timeout=1)
    except serial.SerialException as exc:
        print(f"Erreur d'ouverture du port {args.port}: {exc}")
        sys.exit(1)

    saved_count = 0
    skipped_count = 0

    print(f"Ecoute sur {args.port}... (duree fixe: {args.duration}s, "
          f"Ctrl+C pour arreter plus tot)")
    print(f"Experience: {args.experiment_id} | Zone: {args.zone} | "
          f"Scenario: {args.scenario} | Distance: {args.distance} m")
    print(f"Sauvegarde vers: {output_path}")

        # Delay avant démarrage (pour te laisser sortir de la scène)
    START_DELAY_S = 10

    print(f"\nDemarrage dans {START_DELAY_S} secondes...")

    for i in range(START_DELAY_S, 0, -1):
        print(f"  -> {i}...")
        time.sleep(1)

    print(">>> Capture en cours ! <<<\n")

    start_time = time.monotonic()
    end_time = start_time + args.duration
    last_progress_print = 0.0

    try:
        with open(output_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp_us", "rssi", "channel", "mac", "csi_len", "csi_data",
                "experiment_id", "zone_id", "scenario",
                "distance_tx_rx_m", "ground_truth", "comment","marker_state"
            ])

            while True:
                now = time.monotonic()
                if now >= end_time:
                    print(f"\nDuree de {args.duration}s atteinte, arret automatique.")
                    break

                # Affichage du temps restant toutes les ~10 secondes,
                # sans dependre du debit de mesures recues (utile si le
                # flux CSI est faible ou interrompu).
                if now - last_progress_print >= 10:
                    remaining = end_time - now
                    print(f"[{format_remaining(remaining)} restant] "
                          f"{saved_count} mesures sauvegardees...")
                    last_progress_print = now

                try:
                    raw_line = ser.readline().decode(errors="ignore").strip()
                except serial.SerialException as exc:
                    print(f"Erreur de lecture serie: {exc}")
                    break

                if not raw_line.startswith("CSV,"):
                    continue

                fields = raw_line.split(",")

                # Une ligne valide a exactement 7 champs :
                # ["CSV", timestamp, rssi, channel, mac, csi_len, csi_data]
                if len(fields) != 8:
                    skipped_count += 1
                    continue

                row = fields[1:] + [
                    args.experiment_id,
                    args.zone,
                    args.scenario,
                    args.distance,
                    ground_truth,
                    args.comment,
                ]
                writer.writerow(row)
                saved_count += 1

                if saved_count % 50 == 0:
                    file.flush()

    except KeyboardInterrupt:
        print("\nArret demande par l'utilisateur (avant la fin de la duree prevue).")
    finally:
        ser.close()
        elapsed = time.monotonic() - start_time
        print(f"\nTermine. Duree reelle: {elapsed:.1f}s | "
              f"{saved_count} mesures sauvegardees, "
              f"{skipped_count} lignes ignorees (corrompues).")
        print(f"Fichier: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()