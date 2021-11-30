import numpy as np
import cmath
import math
import pandas as pd
t1 = np.array([[11, 24000], [33, 200000], [66, 600000], [110, 11000000], [
    132, 20000000], [166, 35000000], [230, 90000000]])

t2 = np.array([[11, 1], [33, 2], [66, 4], [110, 5], [
    132, 6], [166, 8], [230, 10]])

wire_area_t = np.array([[0.1935, 100],
                        [0.2580, 127],
                        [0.3225, 148],
                        [0.3870, 170],
                        [0.4515, 190],
                        [0.5160, 210],
                        [0.5805, 230],
                        [0.6450, 255],
                        [0.9675, 350],
                        [1.2900, 425],
                        [1.6125, 505],
                        [1.9350, 580],
                        [2.2575, 655],
                        [2.5800, 715],
                        [2.9025, 775],
                        [3.2250, 825]])


def get_ll_voltage(transmitting_power, kilometer):
    kw_km = transmitting_power*kilometer
    idx = next(x for x, val in enumerate(t1[:, 1])
               if val > kw_km)
    return idx


def get_Dm(kv_idx):
    Dm = t2[kv_idx, 1]
    return Dm


def calculate_Ir(transmitting_power, kv_idx, pf):
    Ir = transmitting_power/(np.sqrt(3)*t1[kv_idx, 0]*pf)
    Ir_angle = np.arccos(pf)
    Ir = complex(Ir*np.cos(Ir_angle), -Ir*np.sin(Ir_angle))
    return Ir


def get_wire_area(Ir):
    idx_ir = next(x for x, val in enumerate(wire_area_t[:, 1])
                  if val > Ir)
    return idx_ir


acsr_table = np.array([
    [0.161, 6, 0.236, 1, 0.236, 0.708, 1.0891, 106.2, 954.8],
    [0.322, 6, 0.335, 1, 0.335, 1.005, 0.5400, 214.0, 1864.3],
    [0.387, 6, 0.365, 1, 0.365, 1.097, 0.4550, 255, 2204.5],
    [0.484, 6, 0.409, 1, 0.409, 1.227, 0.3640, 318, 2742.0],
    [0.645, 6, 0.472, 1, 0.157, 1.417, 0.2720, 395, 3311.2],
    [0.805, 30, 0.236, 7, 0.236, 1.654, 0.2200, 605, 5764.0],
    [0.968, 30, 0.259, 7, 0.259, 1.814, 0.1832, 728, 6883.0],
    [1.125, 30, 0.279, 7, 0.279, 1.956, 0.1572, 847, 7953.0],
    [1.290, 30, 0.299, 7, 0.299, 2.013, 0.1370, 975, 9098.0],
    [1.613, 30, 0.335, 7, 0.335, 2.347, 0.1091, 1218, 11306.0]
])

GMR_table = np.array([[7, 0.726],
                      [19, 0.758],
                      [37, 0.768],
                      [61, 0.772],
                      [91, 0.774],
                      [127, 0.776]])


def get_acsr_specs(idx_ir):
    required_wire_area = wire_area_t[idx_ir, 0]
    idx_acsr = next(x for x, val in enumerate(acsr_table[:, 0])
                    if val > required_wire_area)
    acsr_specs = np.array([acsr_table[idx_acsr, 0], acsr_table[idx_acsr, 1], acsr_table[idx_acsr, 2], acsr_table[idx_acsr, 3],
                           acsr_table[idx_acsr, 4], acsr_table[idx_acsr, 5], acsr_table[idx_acsr, 6], acsr_table[idx_acsr, 7], acsr_table[idx_acsr, 8]])
    return acsr_specs


def get_total_resistance(kilometer, acsr_specs):
    total_resistance = acsr_specs[6]*kilometer
    return total_resistance


def get_total_strands(acsr_specs):
    total_strands = acsr_specs[1]+acsr_specs[3]
    return total_strands


def get_GMR(total_strands, acsr_specs):
    r = (acsr_specs[5])/2
    idx_GMR = next(x for x, val in enumerate(GMR_table[:, 0])
                   if val == total_strands)
    GMR = GMR_table[idx_GMR, 1]
    GMR = GMR*r
    return GMR


def get_line_inductance(GMR, Dm):
    Inductance = 2*(10**-7)*np.log((Dm/(GMR/100)))
    return Inductance


def get_inductive_reactance(Inductance, frequency, kilometer):
    Xl = Inductance*2*np.pi*frequency*kilometer*1000
    return Xl


def get_capacitance_reactance(Dm, acsr_specs, kilometer):
    r = (acsr_specs[5]/2)
    # print(r)
    Capacitance = (2*np.pi*8.854*1000*kilometer)/(1000000000000*np.log((Dm/r)))
    return Capacitance


def Impedance(total_resistance, Xl):
    Z = complex(total_resistance, Xl)
    return Z


def Susceptance(Capacitance, frequency):
    Y = 2*np.pi*frequency*Capacitance
    Y = complex(0, Y)
    return Y


def get_ABCD(Z, Y):
    A = 1 + (Y*Z)/2
    D = A
    B = Z
    C = Y*((1 + (Y*Z)/4))
    ABCD = np.array([A, B, C, D])
    return ABCD


"""def receiving_current(idx, transmitting_power, pf):
    Ir = transmitting_power/(np.sqrt(3)*t1[idx, 0]*pf)
    return """


def get_Vs_Is(ABCD, idx, Ir):
    Vr = (t1[idx, 0]*1000)/np.sqrt(3)
    A = ABCD[0]
    B = ABCD[1]
    C = ABCD[2]
    D = ABCD[3]
    Vs = A*Vr + B*Ir
    Vs = abs(Vs)
    return Vs


def get_VR(Vs_Is, idx):
    A = abs(ABCD[0])
    # print(A)
    Vr_phase = Vs_Is/A
    # print(Vr_phase)
    Vr_line = Vr_phase*np.sqrt(3)
    # print(Vr_line)
    Vr = t1[idx, 0]*1000
    VR = ((Vr_line - Vr)/Vr)*100
    return VR


def calculate_Vd0_Vph(acsr_specs, Dm, idx):
    Vd0 = 21.1*m*(acsr_specs[5]/2)*delta*np.log(Dm/(acsr_specs[5]*100/2))
    Vph_Vd0 = t1[idx, 0]/Vd0
    return Vph_Vd0


K_table = np.array([[0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
                    [0.012, 0.018, 0.05, 0.08, 0.3, 1.0, 3.5, 6.0, 8.0]])


def calculate_corona(frequency, idx, Dm, acsr_specs, Vph_Vd0):
    idx_K = next(x for x, val in enumerate(K_table[0, :])
                 if val > Vph_Vd0)
    Pc = ((21*frequency*t1[idx, 0]*t1[idx, 0]) /
          (1000000*np.log10(Dm/(acsr_specs[5]*100/2))*np.log10(Dm/(acsr_specs[5]*100/2))))*K_table[1, idx_K]
    return Pc


'''transmitting_power = float(input("Enter transmitting power in KW"))
kilometer = float(input("Enter distance in kilometers"))
frequency = float(input("Enter frequency"))
pf = float(input("Enter power factor"))
VR_percent = float(input("Enter voltage regulation range"))
corona_loss_range = float(input("Input maximum corona loss"))'''
transmitting_power = 88000
kilometer = 160
frequency = 50
pf = 0.88
VR_percent = 15
corona_loss_range = 0.6

final_data = []
m = 0.82
delta = 1

Vid = get_ll_voltage(transmitting_power, kilometer)


VR = 100

while(VR_percent < VR):

    Dm = get_Dm(Vid)
    Ir = calculate_Ir(transmitting_power, Vid, pf)
    Ir_mag = cmath.polar(Ir)[0]
    wire_area = get_wire_area(Ir_mag)
    acsr_specs = get_acsr_specs(wire_area)
    total_resistance = get_total_resistance(kilometer, acsr_specs)
    total_strands = get_total_strands(acsr_specs)
    GMR = get_GMR(total_strands, acsr_specs)
    L = get_line_inductance(GMR, Dm)
    Xl = get_inductive_reactance(L, frequency, kilometer)
    Xc = get_capacitance_reactance(Dm, acsr_specs, kilometer)
    Z = Impedance(total_resistance, Xl)
    Y = Susceptance(Xc, frequency)
    ABCD = get_ABCD(Z, Y)
    Vs_Is = get_Vs_Is(ABCD, Vid, Ir)
    VR = get_VR(Vs_Is, Vid)
    Vph_Vd0 = calculate_Vd0_Vph(acsr_specs, Dm, Vid)
    Corona = calculate_corona(frequency, Vid, Dm, acsr_specs, Vph_Vd0)

    Nominal_copper_area = acsr_specs[0]
    Number_of_Aluminium_strands_selected = acsr_specs[1]
    Aluminium_Dia = acsr_specs[2]
    Number_of_copper_strands_selected = acsr_specs[3]
    Copper_Dia = acsr_specs[4]
    Overall_Dia = acsr_specs[5]
    Resistance_per_km = acsr_specs[6]
    Weight_per_km = acsr_specs[7]
    Breaking_Load = acsr_specs[8]

    data = [t1[Vid, 0], Ir_mag, total_resistance, Nominal_copper_area, Number_of_Aluminium_strands_selected, Aluminium_Dia, Number_of_copper_strands_selected,
            Copper_Dia, Overall_Dia, Resistance_per_km, Weight_per_km,  Breaking_Load, total_strands, GMR, Xl, Xc, Vs_Is, VR, Corona]
    final_data.append(data)
    Vid = Vid + 1


df = pd.DataFrame(final_data, columns=['Transmission Voltage Selected', 'Transmission current',
                                       'Resistance of transmission line', 'Nominal copper area', 'Number of Aluminium strands selected', 'Aluminium Dia', 'Number of copper strands selected', 'Copper Dia', 'Overall Dia', 'Resistance per km', 'Weight per km', 'Breaking Load',  'Total Number of strands', 'GMR', 'Inductive Reactance', 'Capacitive Reactance', 'Receiving Voltage', 'Voltage Regulation', 'Corona Loss'])
final_data2 = []
Vid = Vid - 1
if (Corona > corona_loss_range):
    wire_area = get_wire_area(Ir_mag) + 1
    while(Corona > corona_loss_range):

        acsr_specs = get_acsr_specs(wire_area)
        total_resistance = get_total_resistance(kilometer, acsr_specs)
        total_strands = get_total_strands(acsr_specs)
        GMR = get_GMR(total_strands, acsr_specs)
        L = get_line_inductance(GMR, Dm)
        Xl = get_inductive_reactance(L, frequency, kilometer)
        Xc = get_capacitance_reactance(Dm, acsr_specs, kilometer)
        Z = Impedance(total_resistance, Xl)
        Y = Susceptance(Xc, frequency)
        ABCD = get_ABCD(Z, Y)
        Vs_Is = get_Vs_Is(ABCD, Vid, Ir)
        VR = get_VR(Vs_Is, Vid)
        Vph_Vd0 = calculate_Vd0_Vph(acsr_specs, Dm, Vid)
        Corona = calculate_corona(frequency, Vid, Dm, acsr_specs, Vph_Vd0)

        Nominal_copper_area = acsr_specs[0]
        Number_of_Aluminium_strands_selected = acsr_specs[1]
        Aluminium_Dia = acsr_specs[2]
        Number_of_copper_strands_selected = acsr_specs[3]
        Copper_Dia = acsr_specs[4]
        Overall_Dia = acsr_specs[5]
        Resistance_per_km = acsr_specs[6]
        Weight_per_km = acsr_specs[7]
        Breaking_Load = acsr_specs[8]

        data2 = [t1[Vid, 0], Ir_mag, total_resistance, Nominal_copper_area, Number_of_Aluminium_strands_selected, Aluminium_Dia, Number_of_copper_strands_selected,
                 Copper_Dia, Overall_Dia, Resistance_per_km, Weight_per_km,  Breaking_Load, total_strands, GMR, Xl, Xc, Vs_Is, VR, Corona]
        wire_area = wire_area + 1
        final_data2.append(data2)

df2 = pd.DataFrame(final_data2, columns=['Transmission Voltage Selected', 'Transmission current',
                                         'Resistance of transmission line', 'Nominal copper area', 'Number of Aluminium strands selected', 'Aluminium Dia', 'Number of copper strands selected', 'Copper Dia', 'Overall Dia', 'Resistance per km', 'Weight per km', 'Breaking Load',  'Total Number of strands', 'GMR', 'Inductive Reactance', 'Capacitive Reactance', 'Receiving Voltage', 'Voltage Regulation', 'Corona Loss'])


df.to_csv('design_sheet_1.csv')
df2.to_csv('design_sheet_2.csv')
print(df)
print(df2)
