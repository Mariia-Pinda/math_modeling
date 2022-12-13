from collections import defaultdict
import random
import numpy as np
import simpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk

CASHIER_LINES = 6
random.seed(42)
ARRIVALS = [random.expovariate(6) for _ in range(4000)]
CUSTOMER_COUNT = [1 for _ in range(4000)]

arrivals = defaultdict(lambda: 0)
cashier_waits = defaultdict(lambda: [])
event_log = []


def register_arrivals(time, num):
    arrivals[int(time)] += num


def register_cashier_wait(time, wait):
    cashier_waits[int(time)].append(wait)


def avg_wait(raw_waits):
    waits = [w for i in raw_waits.values() for w in i]
    return round(np.mean(waits), 1) if len(waits) > 0 else 0


def max_wait(raw_waits):
    waits = [w for i in raw_waits.values() for w in i]
    return round(max(waits), 1) if len(waits) > 0 else 0


def register_customer_arrival(time, customer_id, people_created):
    register_arrivals(time, len(people_created))
    print(f"Customer #{customer_id} arrived at {time}")
    event_log.append({
        "event": "CUSTOMER_ARRIVAL",
        "time": round(time, 2),
        "customerId": customer_id,
        "peopleCreated": people_created
    })


def register_visitor_moving_to_cashier(person, cashier_line, queue_begin, queue_end, service_begin, service_end):
    wait = queue_end - queue_begin
    service_time = service_end - service_begin
    register_cashier_wait(queue_end, wait)
    print(f"Servicing customer waited {wait} minutes in Line {cashier_line}, needed {service_time} minutes to complete")
    event_log.append({
        "event": "WAIT_IN_CASHIER_LINE",
        "person": person,
        "cashierLine": cashier_line,
        "time": round(queue_begin, 2),
        "duration": round(queue_end - queue_begin, 2)
    })
    event_log.append({
        "event": "PAY_PURCHASES",
        "person": person,
        "cashierLine": cashier_line,
        "time": round(service_begin, 2),
        "duration": round(service_end - service_begin, 2)
    })


main = tk.Tk()
main.title("Modeling the Work of Cashiers")
main.config(bg="#fff")
top_frame = tk.Frame(main)
top_frame.pack(side=tk.TOP, expand=False)
canvas = tk.Canvas(main, width=1300, height=350, bg="white")
canvas.pack(side=tk.TOP, expand=False)

f = plt.Figure(figsize=(2, 2), dpi=72)
a3 = f.add_subplot(121)
a3.plot()
a1 = f.add_subplot(222)
a1.plot()
a2 = f.add_subplot(224)
a2.plot()
data_plot = FigureCanvasTkAgg(f, master=main)
data_plot.get_tk_widget().config(height=400)
data_plot.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


class QueueGraphics:
    text_height = 30
    icon_top_margin = -8

    def __init__(self, icon_file, icon_width, queue_name, num_lines, canvas, x_top, y_top):
        self.icon_file = icon_file
        self.icon_width = icon_width
        self.queue_name = queue_name
        self.num_lines = num_lines
        self.canvas = canvas
        self.x_top = x_top
        self.y_top = y_top

        self.image = tk.PhotoImage(file=self.icon_file)
        self.icons = defaultdict(lambda: [])
        for i in range(num_lines):
            canvas.create_text(x_top, y_top + (i * self.text_height), anchor=tk.NW, text=f"{queue_name} #{i + 1}")
        self.canvas.update()

    def add_to_line(self, cashier_number):
        count = len(self.icons[cashier_number])
        x = self.x_top + 60 + (count * self.icon_width)
        y = self.y_top + ((cashier_number - 1) * self.text_height) + self.icon_top_margin
        self.icons[cashier_number].append(
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.image)
        )
        self.canvas.update()

    def remove_from_line(self, cashier_number):
        if len(self.icons[cashier_number]) == 0: return
        to_del = self.icons[cashier_number].pop()
        self.canvas.delete(to_del)
        self.canvas.update()


def cashiers(canvas, x_top, y_top):
    return QueueGraphics("person-resized.gif", 18, "Cashier", CASHIER_LINES, canvas, x_top, y_top)


class ClockAndData:
    def __init__(self, canvas, x1, y1, x2, y2, time):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.canvas = canvas
        self.train = canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, fill="#fff")
        self.time = canvas.create_text(self.x1 + 10, self.y1 + 10, text="Time = " + str(round(time, 1)) + "m",
                                       anchor=tk.NW)
        self.avg_wait = canvas.create_text(self.x1 + 10, self.y1 + 40,
                                              text="Avg. Wait  = " + str(avg_wait(cashier_waits)), anchor=tk.NW)
        self.max_wait = canvas.create_text(self.x1 + 10, self.y1 + 70,
                                            text="Max Wait = " + str(max_wait(cashier_waits)), anchor=tk.NW)
        self.canvas.update()

    def tick(self, time):
        self.canvas.delete(self.time)
        self.canvas.delete(self.avg_wait)
        self.canvas.delete(self.max_wait)

        self.time = canvas.create_text(self.x1 + 10, self.y1 + 10, text="Time = " + str(round(time, 1)) + "m",
                                       anchor=tk.NW)
        self.avg_wait = canvas.create_text(self.x1 + 10, self.y1 + 30,
                                              text="Avg. Wait  = " + str(avg_wait(cashier_waits)) + "m",
                                              anchor=tk.NW)
        self.max_wait = canvas.create_text(self.x1 + 10, self.y1 + 50,
                                            text="Max Wait = " + str(max_wait(cashier_waits)) + "m", anchor=tk.NW)

        a1.cla()
        a1.set_xlabel("Time")
        a1.set_ylabel("Avg. Wait (minutes)")
        a1.step([t for (t, waits) in cashier_waits.items()], [np.mean(waits) for (t, waits) in cashier_waits.items()])

        a2.cla()
        a2.set_xlabel("Time")
        a2.set_ylabel("Max Wait (minutes)")
        a2.step([t for (t, waits) in cashier_waits.items()], [max(waits) for (t, waits) in cashier_waits.items()])

        a3.cla()
        a3.set_xlabel("Time")
        a3.set_ylabel("Arrivals")
        a3.bar([t for (t, a) in arrivals.items()], [a for (t, a) in arrivals.items()])

        data_plot.draw()
        self.canvas.update()


cashiers_create = cashiers(canvas, 340, 20)
clock = ClockAndData(canvas, 1100, 260, 1290, 340, 0)


def pick_shortest(lines):
    shuffled = list(zip(range(len(lines)), lines))  # tuples of (i, line)
    random.shuffle(shuffled)
    shortest = shuffled[0][0]
    for i, line in shuffled:
        if len(line.queue) < len(lines[shortest].queue):
            shortest = i
            break
    return (lines[shortest], shortest + 1)


def create_clock(env):
    while True:
        yield env.timeout(0.1)
        clock.tick(env.now)


def customer_arrival(env, cashier_lines):
    next_person_id = 0
    while True:
        next_customer = ARRIVALS.pop()
        customer_count = CUSTOMER_COUNT.pop()

        yield env.timeout(next_customer)
        people_ids = list(range(next_person_id, next_person_id + customer_count))
        register_customer_arrival(env.now, next_person_id, people_ids)
        next_person_id += customer_count

        while len(people_ids) > 0:
            remaining = len(people_ids)
            group_size = min(round(random.gauss(1, 0)), remaining)
            people_processed = people_ids[-group_size:]
            people_ids = people_ids[:-group_size]
            env.process(serving_customer(env, people_processed, cashier_lines))


def serving_customer(env, people_processed, cashier_lines):
    queue_begin = env.now
    cashier_line = pick_shortest(cashier_lines)
    with cashier_line[0].request() as req:
        for _ in people_processed: cashiers_create.add_to_line(cashier_line[1])
        yield req
        for _ in people_processed: cashiers_create.remove_from_line(cashier_line[1])
        queue_end = env.now

        for person in people_processed:
            service_begin = env.now
            yield env.timeout(random.expovariate(11/10))
            service_end = env.now
            register_visitor_moving_to_cashier(person, cashier_line[1], queue_begin, queue_end,
                                               service_begin, service_end)


env = simpy.Environment()

scanner_lines = [simpy.Resource(env, capacity=1) for _ in range(CASHIER_LINES)]

env.process(customer_arrival(env, scanner_lines))
env.process(create_clock(env))
env.run(until=1440)
