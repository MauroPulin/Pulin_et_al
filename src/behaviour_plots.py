from custom_bhv_plots import *

data_path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Mauro_Pulin\cicada_preprocessed_results'
saving_path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Mauro_Pulin\bhv_plots'

# plot_average_across_days(data_path, 'whisker_learning_table.csv', saving_path)
# plot_single_session(data_path, saving_path)
# plot_single_mouse_across_days(data_path, saving_path)
# plot_average_reaction_time_across_days(data_path, saving_path)
# plot_single_mouse_reaction_time_days_0_1(data_path, saving_path)
# plot_single_mouse_reaction_time_across_days(data_path, saving_path)
# plot_single_session_reaction_time(data_path, saving_path)
plot_particle_test(data_path, 'particle_test_table.csv', saving_path)