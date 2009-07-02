package sibtra.util;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

/**
 * Spinner que automáticamente invoca el set para modificar parámetro asociado.
 * @author alberto
 *
 */
public class SpinnerDouble extends JSpinner implements ChangeListener {
	
	private Object objeto;
	private Method metodo;
	private SpinnerNumberModel snm;

	public SpinnerDouble(Object obj, String nomMetodo
			, double min, double max, double paso, double valIni) {
		construye(obj,nomMetodo,min,max,paso,valIni);
	}

	/** Constructor con valor inicial leido con método get correspondiente al set pasado */
	public SpinnerDouble(Object obj, String nomMetodo, double min, double max, double paso) {
		if(obj==null)
			throw new IllegalArgumentException("El objeto pasado ha de ser != de null");
		String nomGet="";
		try {
			nomGet="get"+nomMetodo.substring(3);
			Method metGet=obj.getClass().getMethod(nomGet,(Class[])null);
			construye(obj,nomMetodo,min,max,paso,(Double)(metGet.invoke(obj, (Object[])null)));
		} catch (SecurityException e) {
			throw new IllegalArgumentException("El metodo "+nomGet
					+ " no se puede invocar por razones de seguridad");
		} catch (NoSuchMethodException e) {
			throw new IllegalArgumentException("El metodo "+nomGet
					+ " no existe sin argumentos en la calse "
					+ obj.getClass().getCanonicalName());			
		} catch (IllegalArgumentException e) {
			// TODO Bloque catch generado automáticamente
			setEnabled(false);
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Bloque catch generado automáticamente
			setEnabled(false);
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			// TODO Bloque catch generado automáticamente
			setEnabled(false);
			e.printStackTrace();
		}
	}

	/** Constructor con paso=1, y valor inicial leido con método get */
	public SpinnerDouble(Object obj, String nomMetodo, double min, double max) {
		this(obj,nomMetodo,min,max,1);
	}

	private void construye(Object obj, String nomMetodo
			, double min, double max, double paso, double valIni) {
		this.objeto=obj;
		if(obj==null)
			throw new IllegalArgumentException("El objeto pasado ha de ser != de null");
		try {
			Class[] retorno={double.class};
			this.metodo=objeto.getClass().getMethod(nomMetodo,retorno);
		} catch (SecurityException e) {
			throw new IllegalArgumentException("El metodo "+nomMetodo
					+ " no se puede invocar por razones de seguridad");
		} catch (NoSuchMethodException e) {
			throw new IllegalArgumentException("El metodo "+nomMetodo
					+ " no existe con único argumento double en clase "
					+ objeto.getClass().getCanonicalName());			
		}
		snm=new SpinnerNumberModel(valIni,min,max,paso);
		snm.addChangeListener(this);
		setModel(snm);
		setEnabled(true);
	}

	public void stateChanged(ChangeEvent ce) {
		if(ce.getSource()==snm) {
			//invocamos método para fijar el nuevo valor
			try {
				Object[] argumento={new Double(snm.getNumber().doubleValue())};
				metodo.invoke(objeto, argumento);
			} catch (IllegalArgumentException e) {
				// TODO Bloque catch generado automáticamente
				setEnabled(false);
				e.printStackTrace();
			} catch (IllegalAccessException e) {
				// TODO Bloque catch generado automáticamente
				setEnabled(false);
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				// TODO Bloque catch generado automáticamente
				setEnabled(false);
				e.printStackTrace();
			}

		}
	}


}
