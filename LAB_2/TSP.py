import math
import random
import matplotlib.pyplot as plt


#  Rajasthan tourist locations

locations = {

    "Jaipur - Hawa Mahal": (26.9239, 75.8267),

    "Neemrana Fort": (27.9922, 76.3857),

    "Alwar City Palace": (27.5625, 76.6250),

    "Bharatpur Bird Sanctuary": (27.2173, 77.4901),

    "Dholpur": (26.6929, 77.8797),

    "Ranthambore": (26.0173, 76.5026),

    "Karauli": (26.4500, 77.0500),

    "Bundi": (25.4385, 75.6374),

    "Kota": (25.1800, 75.8350),

    "Chittorgarh Fort": (24.8896, 74.6240),

    "Udaipur": (24.5858, 73.7135),

    "Banswara": (23.5411, 74.4425),

    "Mount Abu": (24.5926, 72.7156),

    "Jodhpur": (26.4489, 73.0064),

    "Barmer": (25.7500, 71.4000),

    "Jaisalmer": (27.0524, 70.8613),

    "Bikaner": (28.0176, 73.3150),

    "Mandawa": (28.0559, 75.1495),

    "Ajmer": (26.4521, 74.6387),

    "Pushkar": (26.4902, 74.5521)
}

#  Haversine distance

def haversine(coord1, coord2):

    R = 6371  

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + \
        math.cos(lat1) * math.cos(lat2) * \
        math.sin(dlon / 2) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

#  Create distance matrix

names = list(locations.keys())
n = len(names)

distance_matrix = [[0] * n for _ in range(n)]

for i in range(n):
    for j in range(n):
        distance_matrix[i][j] = haversine(
            locations[names[i]],
            locations[names[j]]
        )

#  Calculate total cost of a tour

def tour_cost(tour):

    total = 0

    for i in range(n - 1):
        total += distance_matrix[tour[i]][tour[i + 1]]

    total += distance_matrix[tour[-1]][tour[0]]

    return total

#  Generate neighbouring solution

def get_neighbor(tour):

    new_tour = tour.copy()

    i, j = random.sample(range(n), 2)

    # Reverse the section between i and j
    if i > j:
        i, j = j, i

    new_tour[i:j + 1] = reversed(new_tour[i:j + 1])

    return new_tour

#  Simulated Annealing

def simulated_annealing():

    current_tour = list(range(n))
    random.shuffle(current_tour)

    current_cost = tour_cost(current_tour)

    best_tour = current_tour.copy()
    best_cost = current_cost

    temperature = 10000
    cooling_rate = 0.995
    minimum_temperature = 0.1

    while temperature > minimum_temperature:

        new_tour = get_neighbor(current_tour)

        new_cost = tour_cost(new_tour)

        delta = new_cost - current_cost

        if delta < 0:

            current_tour = new_tour
            current_cost = new_cost

        else:

            probability = math.exp(-delta / temperature)

            if random.random() < probability:
                current_tour = new_tour
                current_cost = new_cost

        if current_cost < best_cost:

            best_tour = current_tour.copy()
            best_cost = current_cost

        temperature *= cooling_rate

    return best_tour, best_cost

#  Run Simulated Annealing

overall_best_tour = None
overall_best_cost = float('inf')

for run in range(10):

    tour, cost = simulated_annealing()

    print("Run", run + 1, ":", round(cost, 2), "km")

    if cost < overall_best_cost:
        overall_best_cost = cost
        overall_best_tour = tour.copy()


best_tour = overall_best_tour
best_cost = overall_best_cost

print("\n")
print("BEST TOUR")
print("\n")

for i in best_tour:
    print(names[i], "->")

print(names[best_tour[0]])

print("\nBest Total Distance:",
      round(best_cost, 2), "km")

#  Display result

print("\nBest Tour Found:\n")

for i in best_tour:
    print(names[i], "->")

print(names[best_tour[0]])

print("\nTotal Distance:",
      round(best_cost, 2), "km")

# Plot the final Rajasthan tour with place names

plt.figure(figsize=(16, 10))

route_x = []
route_y = []

for i in best_tour:
    lat, lon = locations[names[i]]
    route_x.append(lon)
    route_y.append(lat)

lat, lon = locations[names[best_tour[0]]]
route_x.append(lon)
route_y.append(lat)

plt.plot(
    route_x,
    route_y,
    marker='o',
    linewidth=2,
    markersize=7
)


for position, city_index in enumerate(best_tour):

    lat, lon = locations[names[city_index]]

    if names[city_index] == "Ajmer":
        dx, dy = 12, -18

    elif names[city_index] == "Pushkar":
        dx, dy = -55, 12

    else:
        dx, dy = 8, 8

    plt.annotate(
        f"{position + 1}. {names[city_index]}",
        (lon, lat),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold"
    )

plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)

plt.title(
    "Rajasthan Tourist Tour using Simulated Annealing",
    fontsize=17,
    fontweight="bold"
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.show()