import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot2d(matrix, vector, title=None, file=None, type='space'):

    extent = [vector.min(), vector.max(), vector.min(), vector.max()]

    plt.rcParams.update({'font.size': 16})
    plt.figure()
    plt.imshow(matrix, extent=extent)
    plt.gca().invert_yaxis()
    plt.title(title)

    if type == 'space':
        plt.xlabel('x [meters]')
        plt.ylabel('y [meters]')

    if type == 'freq':
        plt.xlabel('kx [1 / meters]')
        plt.ylabel('ky [1 / meters]')

    plt.tight_layout()

    if file:
        plt.savefig(file)
        plt.close()
    else:
        plt.show(block=True)

    plt.clf()
    plt.close()
    plt.rcdefaults()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot1d(matrix, vector, title=None, file=None, legend=None):

    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 16})
    plt.figure()

    if isinstance(matrix, list):
        for mat in matrix:
            center = mat.shape[0] // 2
            plt.plot(vector, mat[center, :])
    else:
        center = matrix.shape[0] // 2
        plt.plot(vector, matrix[center, :], linewidth=4)

    plt.xlabel('x [meters]')
    plt.ylabel('normalized intensity [a.u.]')
    plt.title(title)

    if legend:
        plt.legend(legend)

    plt.tight_layout()

    if file:
        plt.savefig(file)
    else:
        plt.show(block=True)

    plt.clf()
    plt.close()
    plt.rcdefaults()